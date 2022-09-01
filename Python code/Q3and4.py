import pandas as pd
import numpy as np
import scipy
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import time
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", 20)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from settings import types, phenotypes, phenotypes_file, eQTLs_file_format, corrected_eQTLs_file_format, \
    QTLs_file_format, corrected_QTLs_file_format, coordinates_filename, \
    genotypes_file_format, preprocessing_out_file_format, gene_coordinates_filename

#######################################################################################
################################# FUNCTIONS FOR eQTL ##################################
#######################################################################################

def filter_genotypes(data_file, genotypes_file):
    data_df = pd.read_csv(data_file)
    genotype_df = pd.read_csv(genotypes_file)
    # keep in genotype_df only the strains that are in data_df:
    first_BXD_index = [idx for idx, col in enumerate(genotype_df.columns) if col.startswith('BXD')][0]
    strains = list(data_df.columns)[first_BXD_index:] # keep only BXD column names, as list to preserve the order
    genotype_df = genotype_df[list(genotype_df.columns)[:first_BXD_index] + strains] # keep the data and relevant strains columns
    # filter neighboring loci:
    tmp = genotype_df[strains].loc[(genotype_df[strains].shift() != genotype_df[strains]).any(axis=1)] # remove consecutive duplicate rows
    print("Done removing consecutive duplicate rows - the reduced size is {} rows.".format(len(tmp)))
    filtered_genotype_df = genotype_df.loc[tmp.index] # Keep only non-duplicative indices.
    print("filtered_genotype_df:\n", filtered_genotype_df)
    #filtered_genotype_df.to_csv('filtered_genotypes_{}.csv'.format(dataset_name), index=False)
    return filtered_genotype_df

def genotypes_to_numeric(genotype_df, dataset_name):
    data_cols = genotype_df.iloc[:, :4]
    strains_cols = genotype_df.iloc[:, 4:]
    strains_cols = strains_cols.replace({'B': 0, 'H': 1, 'D': 2, 'b': 0}) # there was one 'b' in the genotypes file
    genotype_df = pd.concat([data_cols, strains_cols], axis=1)
    genotype_df.to_csv(genotypes_file_format.format(dataset_name), index=False)

def create_numeric_filtered_genotype_file(gene_exp_file, genotypes_file, dataset_name):
    filtered_genotype_df = filter_genotypes(gene_exp_file, genotypes_file)
    genotypes_to_numeric(filtered_genotype_df, dataset_name)


# run the association tests and create dataframe with all p-values.
# gene_exp_df and genotypes_df columns need to be only BXD strains.
def eQTL_association_test(gene_exp_df, genotypes_df, results):
    print("Starting performing association tests")
    i = 0
    sum = 0
    start = time.time()
    for index, g_row in genotypes_df.iterrows():
        i += 1
        results[index] = gene_exp_df.apply(
            lambda ge_row: regression_test(ge_row, g_row),
            axis=1
        )
        if i == 50:
            sum += i
            i = 0
            end = time.time()
            print("done", sum, " time:", (end - start) / 60.0)
            start = time.time()
    return results


def regression_test(gene_exp, genotype): # gene_exp and genotype are rows from the files
    x = np.array(genotype).reshape((-1, 1))
    y = np.array(gene_exp)
    linreg = LinearRegression()
    linreg.fit(x, y)
    y_pred = linreg.predict(x)
    R2 = r2_score(y, y_pred)
    n = len(gene_exp)
    F = R2 / ((1 - R2) / (n - 2))
    p_value = scipy.stats.f.sf(F, 1, n - 2)
    return p_value


def eQTL_create_results_file(gene_exp_file, genotypes_file, dataset_name):
    genotypes_df = pd.read_csv(genotypes_file)
    gene_exp = pd.read_csv(gene_exp_file)
    # create df for the results, with column for each locus:
    results = pd.DataFrame(columns=list(range(0, len(genotypes_df))))
    # run association tests:
    genotypes_frst_BXD_idx = [idx for idx, col in enumerate(genotypes_df.columns) if col.startswith('BXD')][0]
    gene_exp_frst_BXD_idx = [idx for idx, col in enumerate(gene_exp.columns) if col.startswith('BXD')][0]
    results = eQTL_association_test(gene_exp.iloc[:,gene_exp_frst_BXD_idx:], genotypes_df.iloc[:,genotypes_frst_BXD_idx:], results) # start from first BXD
    results.columns = genotypes_df['Locus'] # the loci names
    results.index = list(gene_exp['gene_symbol']) # the genes names
    results.to_csv(eQTLs_file_format.format(dataset_name))
    # create also a corrected p-value results file:
    results = results.mul(results.size)
    results.to_csv(corrected_eQTLs_file_format.format(dataset_name))


def create_loci_positions_df(dataset_name):
    loci_pos_df = pd.read_csv(genotypes_file_format.format(dataset_name))
    relevant_cols = ['Locus', 'Chr_Build37', 'Build37_position']
    loci_pos_df = loci_pos_df[relevant_cols]  # take only these cols
    loci_pos_df.columns = ['locus', 'chr', 'position']  # rename columns for easier use
    loci_pos_df.set_index('locus', inplace=True)
    return loci_pos_df


def create_gene_positions_df():
    gene_pos_df = pd.read_csv(gene_coordinates_filename, delimiter="\t")
    relevant_cols = ['marker symbol', 'representative genome chromosome', 'representative genome start',
                     'representative genome end']
    gene_pos_df = gene_pos_df[relevant_cols]  # take only these cols
    gene_pos_df.columns = ['gene_symbol', 'chromosome', 'start', 'end']  # rename columns for easier use
    gene_pos_df.set_index('gene_symbol', inplace=True)
    return gene_pos_df


def find_eQTLs_cis_and_trans(dataset_name, pvalues_df, save_files=0):
    gene_pos_df = create_gene_positions_df()
    loci_pos_df = create_loci_positions_df(dataset_name)
    loci = list(pvalues_df.columns)
    number_of_tests = pvalues_df.size # for multiple tests correction
    threshold = 0.05/number_of_tests # alpha = 0.05
    twoMega = 2000000
    cis_list = []
    trans_list = []
    cis_pvals = []
    trans_pvals = []
    for index, row in pvalues_df.iterrows(): # index is the name of the gene
        min_pval = min(row)
        if min_pval < threshold: # if there's at least one significant p-value in this row (this gene)
            i = 0
            for pval in row:
                if pval < threshold:
                    eQTL_locus = loci[i]
                    eQTL_chr = loci_pos_df.at[eQTL_locus, 'chr']
                    eQTL_pos = loci_pos_df.at[eQTL_locus, 'position']
                    gene_start = gene_pos_df.at[index, 'start']
                    gene_end = gene_pos_df.at[index, 'end']
                    gene_chr = int(gene_pos_df.at[index, 'chromosome'])
                    new_row = [index, gene_chr, gene_start, gene_end, eQTL_locus, eQTL_chr, eQTL_pos, pval * number_of_tests]
                    if (eQTL_chr == gene_chr) and (abs(eQTL_pos - gene_start) <= twoMega or abs(eQTL_pos - gene_end) <= twoMega):
                        cis_list.append(new_row)
                        cis_pvals.append(pval * number_of_tests) # corrected p-value
                    else:
                        trans_list.append(new_row)
                        trans_pvals.append(pval * number_of_tests) # corrected p-value
                i+=1
    columns = ['gene', 'gene chr', 'gene start', 'gene end', 'eQTL locus', 'eQTL chr', 'eQTL pos', 'corrected pval']
    cis_df = pd.DataFrame(cis_list, columns=columns)
    cis_df['kind'] = 'cis'
    cis_df = cis_df.sort_values(['eQTL chr', 'eQTL pos'])
    trans_df = pd.DataFrame(trans_list, columns=columns)
    trans_df['kind'] = 'trans'
    trans_df = trans_df.sort_values(['eQTL chr', 'eQTL pos'])
    all_eQTLs_df = pd.concat([cis_df, trans_df], ignore_index=True, sort=False)
    if save_files:
        cis_df.to_csv("cis_eQTLs_{}.csv".format(dataset_name), index=False)
        trans_df.to_csv("trans_eQTLs_{}.csv".format(dataset_name), index=False)
        all_eQTLs_df.to_csv("all_eQTLs_{}.csv".format(dataset_name), index=False)
    print("all eQTLs for {} dataset:\n".format(dataset_name), all_eQTLs_df)
    return all_eQTLs_df, cis_pvals, trans_pvals


def plot_pvals_distribution(dataset_name, cis_pvals, trans_pvals):
    plt.figure()
    sns.distplot(cis_pvals, label='Cis-associated', hist=False, rug=True, bins=100)
    sns.distplot(trans_pvals, label='Trans-associated', hist=False, rug=True, bins=100)
    plt.axvline(np.mean(cis_pvals), color='b', linestyle='dashed', linewidth=1)
    plt.axvline(np.mean(trans_pvals), color='orange', linestyle='dashed', linewidth=1)
    _, max_ = plt.ylim()
    plt.text(cis_pvals.mean() + cis_pvals.mean() / 10, max_ - max_ / 10, "Cis mean: {:.5f}".format(cis_pvals.mean()))
    plt.text(trans_pvals.mean() + trans_pvals.mean() / 5, max_ - max_ / 5, "Trans mean: {:.5f}".format(trans_pvals.mean()))
    plt.xlabel('P-value')
    plt.legend()
    plt.savefig('corrected_pvals_distribution_{}.png'.format(dataset_name))
    plt.show()


# count how many genes are associated with each locus and save it in dataframe
def count_associated_genes(genotypes_df, pvalues_df):
    first_BXD_index = [idx for idx, col in enumerate(genotypes_df.columns) if col.startswith('BXD')][0]
    count_df = genotypes_df.iloc[:, :first_BXD_index] # info about the chromosome and position
    threshold = 0.05 / pvalues_df.size # alpha = 0.05
    significant_count = []
    for col in pvalues_df:
        num = len(pvalues_df.loc[pvalues_df[col] < threshold])
        significant_count.append(num)
    count_df['count'] = significant_count
    return count_df


# calculate relative positions of loci in the genome
def calc_positions(genotypes_df):
    pos = [] # relative positions
    chr_len_list = [] # Build37_position in each chromosome, we refer to it as the length of the chromosome
    for i in range(1, 21):
        chr_relevant_rows = genotypes_df.loc[genotypes_df['Chr_Build37'] == i].reset_index()
        last = chr_relevant_rows.at[len(chr_relevant_rows) - 1, 'Build37_position']
        chr_len = last
        chr_len_list.append(chr_len)
        pos += list(chr_relevant_rows['Build37_position'] / chr_len + i)
    return pos, chr_len_list


def plot_hotspots_barplot(dataset_name, count_df):
    pos, _ = calc_positions(count_df)
    count_df['pos'] = pos
    x = count_df['pos']
    y = count_df['count']
    plt.bar(x, y, width=0.02, color='black')
    plt.xticks(range(1, 21))
    plt.xlabel('Chromosome')
    plt.title('eQTL hot spots')
    plt.ylabel('Gene Count')
    plt.savefig('hotspot_bar_plot_{}.png'.format(dataset_name))
    plt.show()


def scatter_plot_hotspots(dataset_name, all_eQTLs_df, genotypes_df):
    _, last_pos_list = calc_positions(genotypes_df) # list of the length of each chromosome, ordered by chromosome number
    # add two new columns for saving the relative positions of the genes and loci:
    all_eQTLs_df['gene loc'] = 0.0  # initiate column
    all_eQTLs_df['eQTL loc'] = 0.0  # initiate column
    for index, row in all_eQTLs_df.iterrows():
        g_chr = row['gene chr']
        e_chr = row['eQTL chr']
        all_eQTLs_df.at[index, 'gene loc'] = (row['gene start'] / last_pos_list[g_chr] + g_chr)
        all_eQTLs_df.at[index, 'eQTL loc'] = (row['eQTL pos'] / last_pos_list[e_chr] + e_chr)
    print("all eQTLs for {} dataset with locus and gene relative positions:\n".format(dataset_name), all_eQTLs_df)

    x_cis = all_eQTLs_df.loc[all_eQTLs_df['kind'] == 'cis']['eQTL loc']
    y_cis = all_eQTLs_df.loc[all_eQTLs_df['kind'] == 'cis']['gene loc']
    x_trans = all_eQTLs_df.loc[all_eQTLs_df['kind'] == 'trans']['eQTL loc']
    y_trans = all_eQTLs_df.loc[all_eQTLs_df['kind'] == 'trans']['gene loc']

    plt.scatter(x_cis, y_cis, color='b', label='Cis')
    plt.scatter(x_trans, y_trans, color='r', s=10, label='Trans')
    plt.xticks(range(1, 21))
    plt.yticks(range(1, 21))
    plt.xlabel('Marker position')
    plt.ylabel('Gene position')
    plt.title('eQTL trans-acting and cis-acting hotspots')
    plt.legend(loc='lower right')
    plt.savefig('scatter_plot_{}.png'.format(dataset_name))
    plt.show()


#######################################################################################
################################## FUNCTIONS FOR QTL ##################################
#######################################################################################

# create a file with the results of the QTL analysis for each of our selected phenotypes
def QTL_create_output(phenotypes_file, genotypes_file, phenotypes, corrected=0):
    pheno_df = pd.read_csv(phenotypes_file)
    pheno_first_BXD_index = [idx for idx, col in enumerate(pheno_df.columns) if col.startswith('BXD')][0]
    pheno_df = pheno_df.iloc[:, pheno_first_BXD_index:]  # keep only the BXD columns
    geno_df = pd.read_csv(genotypes_file)
    geno_first_BXD_index = [idx for idx, col in enumerate(geno_df.columns) if col.startswith('BXD')][0] # for creatong the ouptut later
    pheno_df_strains = pheno_df.columns.values # all available strains in the phenotypes_file
    for phen_loc in phenotypes:
        print("working on phenotype in index {}".format(phen_loc))
        phen = list(pheno_df.loc[phen_loc])  # take only the row of my selected phenotype, as a list
        p_values_for_phen, filtered_geno_df = QTL_association_test(geno_df, phen, pheno_df_strains)
        if corrected: # correct the p-values
            p_values_for_phen = [pval * len(p_values_for_phen) for pval in p_values_for_phen]
        # create the output file:
        output_df = filtered_geno_df.iloc[:, :geno_first_BXD_index]
        output_df[phen_loc] = p_values_for_phen
        print(output_df)
        if corrected:
            output_name = corrected_QTLs_file_format.format(phen_loc)
        else:
            output_name = QTLs_file_format.format(phen_loc)
        output_df.to_csv(output_name, index=False)
        print("Done creating "+output_name)


# do association test between the phenotype and each locus
def QTL_association_test(geno_df, phen, pheno_df_strains):
    values_indices = [i for i, val in enumerate(phen) if ~np.isnan(val)] # the indices of BXDs that have data for the phenotype
    phen = [phen[idx] for idx in values_indices] # keep only non-nan values
    strains = pheno_df_strains[values_indices] # the BXDs that have data for the phenotype
    first_BXD_index = [idx for idx, col in enumerate(geno_df.columns) if col.startswith('BXD')][0]
    geno_df = geno_df[list(geno_df.columns)[:first_BXD_index] + list(strains)] # keep only the BXDs that have data for the phenotype
    # filter neighboring loci:
    tmp = geno_df[strains].loc[(geno_df[strains].shift() != geno_df[strains]).any(axis=1)]  # remove consecutive duplicate rows
    print("Done removing consecutive duplicate rows - the reduced size is {} rows.".format(len(tmp)))
    geno_df = geno_df.loc[tmp.index] # Keep only non-duplicative indices.
    geno_df = geno_df.replace({'B': 0, 'H': 1, 'D': 2, 'b': 0})
    print("filtered and numeric geno df:\n", geno_df)
    p_values_for_phen = []
    for index, row in geno_df.iloc[:,first_BXD_index:].iterrows(): # row is the genotype in a specific locus
        pval = regression_test(phen, list(row))
        p_values_for_phen.append(pval)
    return p_values_for_phen, geno_df


def generate_manhattan_plot(alpha, phen_locs=phenotypes):
    for phen_loc in phen_locs:
        pvalues_df = pd.read_csv(QTLs_file_format.format(phen_loc))
        phen_loc = str(phen_loc)
        pvalues_df[phen_loc] = pvalues_df[phen_loc].apply(pd.to_numeric)
        pvalues_df[phen_loc] = pvalues_df[phen_loc].apply(lambda pval: -np.log10(pval))
        plot = sns.relplot(data=pvalues_df, x='ID_FOR_CHECK', y=phen_loc, aspect=2, s=60,
                           hue='Chr_Build37', palette='bright', legend=None)
        chrom_df = pvalues_df.groupby('Chr_Build37')['ID_FOR_CHECK'].median()
        plot.ax.set_xlabel('Chromosome')
        plot.ax.set_ylabel('-log(P-value)')
        plot.ax.set_xticks(chrom_df)
        plot.ax.set_xticklabels(chrom_df.index)
        plot.fig.suptitle('Manhattan plot for phenotype ' + phen_loc, ha='left')
        thresh = -np.log10(alpha / len(pvalues_df))
        plt.axhline(y=thresh, color='r', linestyle='--', label='Threshold')
        plt.legend()
        plt.savefig('manhattan_plot_{}.png'.format(phen_loc))
        plt.show()


#######################################################################################
#################################### RUN ANALYSIS #####################################
#######################################################################################

def run_QTL_analysis(corrected): # whether to calculate corrected p-values or not
    genotypes_file = coordinates_filename
    QTL_create_output(phenotypes_file, genotypes_file, phenotypes, corrected)

def run_eQTL_analysis(dataset_name):
    gene_exp_file = preprocessing_out_file_format.format(dataset_name)
    raw_genotypes_file = coordinates_filename
    create_numeric_filtered_genotype_file(gene_exp_file, raw_genotypes_file, dataset_name)
    genotypes_file = genotypes_file_format.format(dataset_name) # the function called above creates this file
    eQTL_create_results_file(gene_exp_file, genotypes_file, dataset_name)

def create_plots_for_eQTL_analysis(dataset_name):
    results_file = eQTLs_file_format.format(dataset_name)
    pvalues_df = pd.read_csv(results_file, index_col=0)
    pvalues_df = pvalues_df.apply(pd.to_numeric)
    genotypes_file = genotypes_file_format.format(dataset_name)
    genotypes_df = pd.read_csv(genotypes_file)
    count_df = count_associated_genes(genotypes_df, pvalues_df)
    all_eQTLs_df, cis_pvals, trans_pvals = find_eQTLs_cis_and_trans(dataset_name, pvalues_df)
    plot_pvals_distribution(dataset_name, pd.Series(cis_pvals), pd.Series(trans_pvals))
    plot_hotspots_barplot(dataset_name, count_df)
    scatter_plot_hotspots(dataset_name, all_eQTLs_df, count_df)


# for reporting the number of QTLs found for each phenotype
def count_QTLs(alpha):
    for ploc in phenotypes:
        df = pd.read_csv(corrected_QTLs_file_format.format(ploc))
        ploc = str(ploc)
        QTLs = df.loc[df[ploc] < alpha]
        QTLs_df = QTLs[list(df.columns)[:4]+[ploc]]
        if len(QTLs_df) > 0:
            QTLs_df.to_csv("QTLs_for_phen_{}.csv".format(ploc))
        print("For phen {}:\n".format(ploc), QTLs_df)


if __name__ == '__main__':
    # Q3:
    for type in types:
        run_eQTL_analysis(type)
    for type in types:
        create_plots_for_eQTL_analysis(dataset_name=type)
    # Q4:
    run_QTL_analysis(corrected=0)
    run_QTL_analysis(corrected=1)
    alpha = 0.15
    generate_manhattan_plot(alpha)
    count_QTLs(alpha)




