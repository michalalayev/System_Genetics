import pandas as pd
pd.set_option("display.max_columns", 20)

from settings import types, phenotypes, eQTLs_file_format, corrected_eQTLs_file_format, corrected_QTLs_file_format

alpha = 0.15

def q5():
    f = open("Q5_results.txt", "w")
    for type in types:
        f.write("{} Gene Expression:\n".format(type))
        df = pd.read_csv("all_eQTLs_{}.csv".format(type))
        eQTLs = list(df['eQTL locus'])
        for ploc in phenotypes:
            qtl_df = pd.read_csv(corrected_QTLs_file_format.format(ploc))
            f.write("\nPhenotype {}:\n".format(ploc))
            f.write("QTLs that are also eQTLs:\n")
            QTLs = list(qtl_df.loc[qtl_df[str(ploc)] < alpha]['Locus'])
            common = [snp for snp in QTLs if snp in eQTLs]
            if len(common) == 0:
                f.write("Not found.\n")
            else:
                f.write(', '.join(common) + "\n")
            uncommon = [snp for snp in QTLs if snp not in eQTLs]
            f.write("QTLs that are not eQTLs:\n")
            if len(uncommon) == 0:
                f.write("Not found.\n")
            else:
                f.write(', '.join(uncommon) + "\n")
        f.write("\n")
    f.close()


# create Q5_results.txt:
if __name__ == '__main__':
    q5()



