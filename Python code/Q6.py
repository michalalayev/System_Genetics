import pandas as pd
import functools
import numpy as np
import scipy.stats
import sklearn
from operator import mul
from functools import reduce

from settings import types, phenotypes, phenotypes_file, corrected_eQTLs_file_format, corrected_QTLs_file_format, \
    coordinates_filename, genotypes_file_format, preprocessing_out_file_format, Q6_pairs_file_format, \
    Q6_QTL_max_thrsd, Q6_eQTL_max_thrsd, Q6_model_compute_pval_sample

DEBUG = False
dbg_sample_size = 10

def create_pairs(type,dbg_sample_size=dbg_sample_size, DEBUG=DEBUG):
    eQTLs_df = pd.read_csv(corrected_eQTLs_file_format.format(type), index_col=0, sep=",")
    coordinates_df = pd.read_csv(coordinates_filename, sep=",")
    pairs = []
    for phenotype in phenotypes:
        if DEBUG and dbg_sample_size <= 0:
            break
        QTLs_df = pd.read_csv(corrected_QTLs_file_format.format(phenotype), sep=",", header=0, index_col=1)
        for QTL_locus in QTLs_df.index:
            if DEBUG and dbg_sample_size <= 0:
                break
            QTL_pval = QTLs_df.loc[QTL_locus,str(phenotype)]
            if(QTL_pval <= Q6_QTL_max_thrsd):
                for eQTL_locus in eQTLs_df.columns:
                    if DEBUG and dbg_sample_size <= 0:
                        break
                    if(SNPs_nearby(eQTL_locus,QTL_locus,coordinates_df)):
                        for gene in eQTLs_df.index:
                            if DEBUG and dbg_sample_size<=0:
                                break
                            eQTL_pval = eQTLs_df.loc[gene,eQTL_locus]
                            if(eQTL_pval <= Q6_eQTL_max_thrsd):
                                pairs.append([eQTL_locus,gene,eQTL_pval,QTL_locus,phenotype,QTL_pval])
                                if DEBUG:
                                    dbg_sample_size=dbg_sample_size-1
    pairs_df = pd.DataFrame.from_records(pairs, columns=['eQTL_locus','gene','eQTL_pval','QTL_locus','phenotype','QTL_pval'])
    if pairs_df.empty:
        pairs_df.to_csv(Q6_pairs_file_format.format(type))
        return pairs_df
    # Order the pairs by max(eQTL_pval,QTL_pval)
    # s.t. the best pairs (with minimal pVal) - their model will have a p-value computed for it
    pairs_df['max_pval'] = pairs_df.apply(lambda row: max(row.eQTL_pval,row.QTL_pval),axis=1)
    pairs_df.sort_values(by='max_pval', inplace=True, ascending=True, ignore_index=True)
    pairs_df['comupte_model_pval'] = False
    pairs_df.loc[range(min(Q6_model_compute_pval_sample,pairs_df.shape[0])) ,'comupte_model_pval'] = True
    pairs_df.drop(['max_pval'], inplace=True, axis=1)
    pairs_df.to_csv(Q6_pairs_file_format.format(type))

    return pairs_df

def SNPs_nearby(eQTL_SNP,QTL_SNP,coordinates_df):
    if(eQTL_SNP == QTL_SNP):
        return True
    twoMega = 2000000
    eQTL_chr = coordinates_df.loc[coordinates_df['Locus']==eQTL_SNP, 'Chr_Build37'].iloc[0]
    eQTL_pos = coordinates_df.loc[coordinates_df['Locus']==eQTL_SNP, 'Build37_position'].iloc[0]
    QTL_chr = coordinates_df.loc[coordinates_df['Locus']==QTL_SNP, 'Chr_Build37'].iloc[0]
    QTL_pos = coordinates_df.loc[coordinates_df['Locus']==QTL_SNP, 'Build37_position'].iloc[0]
    return eQTL_chr == QTL_chr and (abs(eQTL_pos - QTL_pos) <= twoMega)

def causality_test(pair,type):
    causality_test_data = extract_test_data(pair,type)
    model = choose_best_model(causality_test_data)
    print("For",type,", for the locus", pair.eQTL_locus, ", the gene", pair.gene, "and the phenotype", pair.phenotype,
          "the most likely model is Model", model)
    if pair['comupte_model_pval']:
        pval = compute_model_pval(causality_test_data,model)
        print("\tModel pVal is {:.3f}".format(pval))
    if pair.eQTL_locus != pair.QTL_locus:
        print("\tMind that", pair.eQTL_locus, "is the eQTL locus. You might also want to consider the nearby QTL locus:",
              pair.QTL_locus)

def extract_test_data(pair, type):
    phenotypes_df = pd.read_csv(phenotypes_file, index_col=0)
    gene_expression_df = pd.read_csv(preprocessing_out_file_format.format(type), sep=",", index_col=0)
    genotypes_df = pd.read_csv(genotypes_file_format.format(type), index_col=1, sep=",")

    first_BXD_index = [idx for idx, col in enumerate(genotypes_df.columns) if col.startswith('BXD')][0]
    strains = list(genotypes_df.columns)[first_BXD_index:] # keep only BXD column names, as list to preserve the order

    bxds_data = []
    for bxd in strains:
        locus = pair.eQTL_locus
        L = genotypes_df.loc[locus,bxd]
        gene = pair.gene
        R = gene_expression_df.loc[gene,bxd]
        phenotype = pair.phenotype
        C = phenotypes_df.loc[phenotype,bxd]
        bxds_data.append([L,R,C])
    df=pd.DataFrame.from_records(bxds_data, columns=['L','R','C'])
    df['R'] = pd.to_numeric(df['R'])
    df['C'] = pd.to_numeric(df['C'])
    df.dropna(subset=df.columns,inplace=True) # Drop null values
    df.reset_index(inplace=True)
    return df

def choose_best_model(data):
    P_L_arr = compute_allele_frequency(data)
    mean_R = data['R'].mean()
    std_R = data['R'].std()
    mean_C = data['C'].mean()
    std_C = data['C'].std()
    rho_RC = data['R'].corr(data['C'])

    mean_R_given_Li_arr = []
    std_R_given_Li_arr = []
    mean_C_given_Li_arr = []
    std_C_given_Li_arr = []
    for allele in range(len(P_L_arr)):
        mean_R_given_Li_arr.append(data.loc[data['L'] == allele, 'R'].mean())
        std_R_given_Li_arr.append(data.loc[data['L'] == allele, 'R'].std())
        mean_C_given_Li_arr.append(data.loc[data['L'] == allele, 'C'].mean())
        std_C_given_Li_arr.append(data.loc[data['L'] == allele, 'C'].std())

    m1_likelihood = 1.0
    m2_likelihood = 1.0
    m3_likelihood = 1.0
    n1 = 0
    n2 = 0
    n3 = 0

    for i,bxd in data.iterrows():
        allele = int(bxd['L'])
        P_Li = P_L_arr[allele]
        Ri_given_Li_mean = mean_R_given_Li_arr[allele]
        Ri_given_Li_std = std_R_given_Li_arr[allele]
        P_Ri_given_Li = scipy.stats.norm(Ri_given_Li_mean,Ri_given_Li_std).pdf(bxd['R'])

        Ci_given_Li_mean = mean_C_given_Li_arr[allele]
        Ci_given_Li_std = std_C_given_Li_arr[allele]
        P_Ci_given_Li = scipy.stats.norm(Ci_given_Li_mean,Ci_given_Li_std).pdf(bxd['C'])

        Ri_given_Ci_mean = mean_R + rho_RC*std_R/std_C * (bxd['R']-mean_C)
        Ri_given_Ci_std = np.square(std_R) * np.sqrt(1-np.square(rho_RC))
        P_Ri_given_Ci = scipy.stats.norm(Ri_given_Ci_mean, Ri_given_Ci_std).pdf(bxd['R'])

        Ci_given_Ri_mean = mean_C + rho_RC*std_C/std_R * (bxd['C']-mean_R)
        Ci_given_Ri_std = np.square(std_C) * np.sqrt(1-np.square(rho_RC))
        P_Ci_given_Ri = scipy.stats.norm(Ci_given_Ri_mean, Ci_given_Ri_std).pdf(bxd['C'])

        m1_bxd_likelihood = P_Li * P_Ri_given_Li * P_Ci_given_Ri
        m2_bxd_likelihood = P_Li * P_Ci_given_Li * P_Ri_given_Ci
        m3_bxd_likelihood = P_Li * P_Ri_given_Li * P_Ci_given_Li

        if(~np.isnan(m1_bxd_likelihood)):
            m1_likelihood = mul(m1_likelihood,m1_bxd_likelihood)
            n1 += 1
        if(~np.isnan(m2_bxd_likelihood)):
            m2_likelihood = mul(m2_likelihood,m2_bxd_likelihood)
            n2 += 1
        if(~np.isnan(m3_bxd_likelihood)):
            m3_likelihood = mul(m3_likelihood,m3_bxd_likelihood)
            n3 += 1

    return get_best_model_by_AICC(m1_likelihood, m2_likelihood, m3_likelihood, n1 ,n2, n3)

def compute_allele_frequency(data):
    freq_0 = 0 if not (0 in data['L'].value_counts()) else data['L'].value_counts()[0]/data.shape[0]
    freq_1 = 0 if not (1 in data['L'].value_counts()) else data['L'].value_counts()[1]/data.shape[0]
    freq_2 = 0 if not (2 in data['L'].value_counts()) else data['L'].value_counts()[2]/data.shape[0]
    return [freq_0,freq_1,freq_2]

def get_best_model_by_AICC(l1, l2, l3, n1, n2, n3):
    AIC1 = compute_corrected_AIC(l1,1+4+3,n1)
    AIC2 = compute_corrected_AIC(l2,1+4+3,n2)
    AIC3 = compute_corrected_AIC(l3,1+4+4,n3)
    AICs = [AIC1,AIC2,AIC3]
    min_value = min(AICs)
    return AICs.index(min_value)+1

# https://en.wikipedia.org/w/index.php?title=Akaike_information_criterion#Modification_for_small_sample_size
def compute_corrected_AIC(L, k, n):
    return 2*k-2*np.log(L) + (2*np.square(k)+2*k)/(n-k-1)

def compute_model_pval(data, model):
    sample_size = 1000
    shuffled_data = data.copy(deep=True)
    model_choices =[0,0,0]
    for i in range(sample_size):
        shuffled_data['R'] = np.random.permutation(shuffled_data['R'].values)
        shuffled_data['C'] = np.random.permutation(shuffled_data['C'].values)
        model_choices[choose_best_model(shuffled_data) - 1] += 1
    return (sample_size-model_choices[model-1])/sample_size

if __name__ == '__main__':
    for type in types:
        create_pairs(type)
        pairs = pd.read_csv(Q6_pairs_file_format.format(type), sep=",", header=0)
        pairs.apply(lambda p: causality_test(p,type), axis=1)