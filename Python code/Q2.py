import pandas as pd
import numpy as np
import time

###################################################################################################
######################################### USE DEBUG MODE? #########################################
###################################################################################################
DEBUG = False
DEBUG_subset = 500

###################################################################################################
####################################### PROGRAM STARTS HERE #######################################
###################################################################################################

''' 
- Merge data file with annotation file to get your input matrix. In your input
  matrix, rows should be annotated with gene identifier (either gene symbol
  or entrez IDs); columns are BXD strain names.
- Remove rows with no gene identifier.
- Remove rows with low maximal value (choose a threshold).
- Remove rows with low variance (choose a threshold).
- The data may contain multiple rows (probes) for the same gene. To have a
  single row per gene, either select one probe while removing the others
  (e.g., select the highest- variance probe) or calculate their average.
'''

from settings import types, gene_expression_filename, gene_symbols_filename, gene_coordinates_filename, \
    preprocessing_out_file_format, low_var_prct_thrsd, low_max_val_prct_thrsd


def preprocessing():
    for type in types:
        dataset_preprocessing(gene_expression_filename, type)

def dataset_preprocessing(filename, type):
    # Load from file
    df = file_to_df(filename, type)
    print("\nFrom file",filename,"reloaded this df for type",type,":\n",df)

    # Find gene symbol and remove genes with no identifier
    df = change_gene_id_to_symbol(df)
    print("\nChanged gene_id to gene symbol:\n", df)
    df = remove_genes_no_identifier(df)
    print("\nRemoved rows with no gene identifier:\n", df)

    # Sort by gene symbol and averaged values for multiple probes
    df = remove_duplicate_genes_and_sort(df)
    print("\nSorted by (unique) ID_REF:\n", df)

    # Add gene positions
    df = add_gene_positions(df)
    print("\nAdded gene positions (genes with no positions were removed):\n",df)

    # Remove rows with low maximal value
    df = remove_rows_by_max_val_thrsd(df, low_max_val_prct_thrsd)
    print("\nRemoved genes with maximal GE lower than the",low_max_val_prct_thrsd,"percentile. The remaining genes:\n",df)

    # Remove rows with low variance
    df = remove_low_var_rows_by_percentage(df, low_var_prct_thrsd)
    print("\nRemoved genes with variance lower than the",low_var_prct_thrsd,"percentile. The remaining genes:\n",df)

    ######################## TEST ########################
    # if (DEBUG):
    #     df.loc[df['ID_REF']=='V1rc8','ID_REF'] = 'V1rb9'
    #
    # Statistics
    # if(DEBUG):
    #     df['min'] = df.iloc[:,4:].min(axis=1, numeric_only=True)
    #     df['avg'] = df.iloc[:,4:].mean(axis=1, numeric_only=True)
    #     df['max'] = df.iloc[:,4:].max(axis=1, numeric_only=True)
    ######################## TEST ########################

    #### NOTE: Mind that filtering by neighbouring loci is done during the eQTL / QTL analysis part ###

    # Write results
    df = df.rename(columns={"ID_REF": "gene_symbol"})
    df.to_csv(preprocessing_out_file_format.format(type), index=False)

def file_to_df(filename,type):
    print("\nFrom file",filename,"loading data for type",type,"...\n")
    batch_to_bxd_df = get_batch_to_bxd_map(filename, type)

    if DEBUG:
        ge_df = pd.read_csv(filename, sep="\t", skiprows=69, nrows=DEBUG_subset)
    else:
        ge_df = pd.read_csv(filename, sep="\t", skiprows=69, skipfooter=1)

    ge_df_of_type = pd.DataFrame({'ID_REF':ge_df['ID_REF']})
    for bxd in sorted(batch_to_bxd_df['BXD'].unique()):
        if(len(batch_to_bxd_df[batch_to_bxd_df['BXD']==bxd].values) > 1):
            print("Found duplicates for BXD:",bxd,'\nValues were averaged')
            ge_df_of_type["BXD"+str(bxd)] = ge_df[batch_to_bxd_df[batch_to_bxd_df['BXD']==bxd].index].mean(axis=1)
        else:
            ge_df_of_type["BXD"+str(bxd)] = ge_df[batch_to_bxd_df[batch_to_bxd_df['BXD']==bxd].index[0]]
    return ge_df_of_type


def get_batch_to_bxd_map(filename, type):
    df = pd.read_csv(filename, sep="\t", skiprows=36, nrows=9)
    # Filter by type
    df = df.loc[:,df.iloc[5,:]==type]
    # BXD number only
    df = df.loc[8,:]
    df = pd.DataFrame({'BXD': df.apply(lambda x: x.split(" ")[2]).map(int)})
    return df

def remove_rows_by_max_val_thrsd(df, pct):
    df['max'] = df.iloc[:, 4:].max(axis=1, numeric_only=True)
    df = df[df['max'] >= np.percentile(df['max'], pct)]
    df = df.drop(columns=['max'])
    return df

def remove_low_var_rows_by_percentage(df, pct):
    df['var'] = df.iloc[:, 4:].var(axis=1, numeric_only=True)
    df = df[df['var'] >= np.percentile(df['var'], pct)]
    df = df.drop(columns=['var'])
    return df

def change_gene_id_to_symbol(df):
    gene_df = pd.read_csv(gene_symbols_filename, sep="\t", skiprows=14, header=0)
    for gene_id in df['ID_REF']:
        found_symbol = gene_df.loc[gene_df['ID'] == gene_id, 'Symbol']
        df.loc[df['ID_REF'] == gene_id, 'ID_REF'] = found_symbol.iloc[0] if len(found_symbol) > 0 else np.NAN
    return df

def remove_genes_no_identifier(df):
    df = df[~df['ID_REF'].isnull()]
    return df

def remove_duplicate_genes_and_sort(df):
    duplicated_genes = df.loc[df['ID_REF'].duplicated() == True,'ID_REF']
    if len(duplicated_genes) > 0:
        print("\nFound these duplicated genes:\n", df.loc[df['ID_REF'].duplicated() == True,'ID_REF'],"\nValues were averaged")
    df = df.groupby('ID_REF').mean().sort_values(by=['ID_REF']).reset_index()
    return df

def add_gene_positions(df):
    coordinates_df = pd.read_csv(gene_coordinates_filename, sep='\t', header=0, dtype={'Chr': str})
    df.insert(1, 'chromosome', "")
    df.insert(2, 'start', np.NAN)
    df.insert(3, 'end', np.NAN)
    for gene_id in df['ID_REF']:
        found = coordinates_df.loc[coordinates_df['marker symbol'] == gene_id]
        chr_col = 'representative genome chromosome'
        start_col = 'representative genome start'
        end_col = 'representative genome end'
        if (len(found) > 0 & len(found[chr_col]) & len(found[start_col]) & len(found[end_col])):
            df.loc[df['ID_REF']==gene_id,'chromosome'] = found[chr_col].iloc[0]
            df.loc[df['ID_REF']==gene_id,'start'] = found[start_col].iloc[0]
            df.loc[df['ID_REF']==gene_id,'end'] = found[end_col].iloc[0]
    df = df[(df['chromosome'] != "") & ~df['chromosome'].isnull() & (df['start'] != "") & ~df['start'].isnull() & (df['end'] != "") & ~df['end'].isnull()]
    return df

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start = time.time()
    preprocessing()
    end = time.time()
    print("Preprocessing is done. Time in seconds: {:.2f}".format(end-start))
