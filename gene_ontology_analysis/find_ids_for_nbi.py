from goatools.anno.genetogo_reader import Gene2GoReader
from goatools.anno.gaf_reader import GafReader
import pandas as pd
import wget
import gzip
import shutil
import os

def download_gene_info():
    if not os.path.exists('gene_info'):
        print("Downloading gene info data from NCBI...")
        url = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_info.gz"
        wget.download(url, "gene_info.gz")
        print("\nExtracting gene info file...")
        
        with gzip.open('gene_info.gz', 'rb') as f_in:
            with open('gene_info', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        os.remove('gene_info.gz')
        print("Gene info file prepared successfully")

def create_mouse_gene_mapping():
    if not os.path.exists('gene_info'):
        download_gene_info()
    
    print("Creating mouse gene mapping...")
    gene_mapping = {}
    mouse_taxid = "10090"
    
    with open('gene_info', 'r') as f:
        next(f)
        for line in f:
            fields = line.strip().split('\t')
            taxid = fields[0]
            if taxid == mouse_taxid:
                gene_id = fields[1]
                symbol = fields[2]
                synonyms = fields[4].split('|') if fields[4] != '-' else []
                
                gene_mapping[symbol.upper()] = gene_id
                
                for syn in synonyms:
                    gene_mapping[syn.upper()] = gene_id

    return gene_mapping

def check_genes_with_mapping(test_genes):
    gene_mapping = create_mouse_gene_mapping()
    
    print(f"\nTotal number of mouse gene symbols in mapping: {len(gene_mapping)}")

    results = []
    for gene in test_genes:
        gene_upper = gene.upper()
        ncbi_id = gene_mapping.get(gene_upper)
        
        results.append({
            'Original_Name': gene,
            'Found_In_DB': ncbi_id is not None,
            'NCBI_Gene_ID': ncbi_id if ncbi_id else 'Not found'
        })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    test_genes = {
        "Abcg2", "Amacr",
        
        "Hist1h1b", "Vim", "Trp63", "Inmt", "Adamtsl2", "Tnc",
        "Fgf18", "Itga8", "Cd38", "Mmp3", "Vegfa", "Gata6",
        "Kcnma1", "Tfrc", "Ager", "F13a1", "Mcpt4", "Foxf2",
        "Epha7", "Hmbs", "E2f8", "Tgfb1", "Nkx2-1", "Cldn5",
        "Gucy1a2", "Prim2", "Tbp", "Sftpd", "Cdh2", "Thy1",
        "Cldn1", "Igfbp3", "Egfr", "Ywhaz", "Hprt", "Abcd1",
        "Nme3", "Mgat4a", "Mmp7", "Hpgds"
    }
    
    try:
        print("Starting gene mapping analysis...")
        results = check_genes_with_mapping(test_genes)
        
        print("\nResults of gene mapping analysis:")
        print(results)
        
        results.to_csv('gene_mapping_results.csv', index=False)
        print("\nDetailed results saved to 'gene_mapping_results.csv'")
        
        print(f"\nTotal genes checked: {len(results)}")
        print(f"Genes found in database: {results['Found_In_DB'].sum()}")
        print(f"Genes not found: {len(results) - results['Found_In_DB'].sum()}")
    
        print("\nFound genes with their NCBI IDs:")
        found_genes = results[results['Found_In_DB']]
        print(found_genes[['Original_Name', 'NCBI_Gene_ID']])
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

#neg_corr_genes = {"AMACR", "ABCG2"}
# all_genes = {
#     "Hist1h1b", "VIM", "P-63", "INMT", "ADAMTSL2", "Tnc", "FGF18", "Shisa3", "integrin subunit alpha 8", "Hist1h2ab", 
#     "CD38", "MMP-3", "Lrp2", "ppia", "THTPA", "Vegf", "GATA-6", "ABCA3", "Kcnma1", "tfrc", "RAGE", "F13A1", "MCPt4",
#     "FOXF2", "EPHA7", "AGER", "hmbs", "E2F8", "TGFB1", "TTF-1", "Claudin5", "GUCY1A2  sGC", "PRIM2", "tbp", "SFTP-D",
#     "N-Cadherin", "Thy1", "Claudin 1", "Igfbp3", "EGFR", "ywhaz", "hprt", "ABCD1", "NME3", "MGAT4A", "MMP7", "HPGDS",
#     "ABCG2", "AMACR"
#     }


