types = ["Myeloid", "Stem"]
phenotypes = [206, 207, 210]
gene_expression_filename = "GSE18067_series_matrix.txt"
gene_symbols_filename = "GPL6238-3838.txt"
gene_coordinates_filename = "MGI_Coordinates.Build37.rpt.txt"
preprocessing_out_file_format = "{}_preprocessed.csv"
phenotypes_file = "phenotypes.csv"
eQTLs_file_format = "eQTL_results_{}.csv"
corrected_eQTLs_file_format = "eQTL_results_corrected_{}.csv"
QTLs_file_format = "QTL_results_{}.csv"
corrected_QTLs_file_format = "QTL_results_corrected_{}.csv"
coordinates_filename = "genotypes.csv"
genotypes_file_format = "numeric_filtered_genotypes_{}.csv"

# Q2 (Preprocessing)
low_max_val_prct_thrsd = 70
low_var_prct_thrsd = 70

#Q6
Q6_QTL_max_thrsd = 0.15
Q6_eQTL_max_thrsd = 0.15
Q6_pairs_file_format = "pairs_{}.csv"
Q6_model_compute_pval_sample = 10
