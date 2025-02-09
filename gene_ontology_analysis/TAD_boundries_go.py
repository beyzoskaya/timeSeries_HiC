import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import zscore
import requests
import json
import os
from distinct_temporal_patterns_go import clean_gene_name
import logging
import matplotlib.pyplot as plt

ENRICHR_URL = "https://maayanlab.cloud/Enrichr"
logging.basicConfig(level=logging.INFO)

def normalize_insulation_scores(df, column_name="Gene1_Insulation_Score"):
    min_val = df[column_name].min()
    max_val = df[column_name].max()
    
    if max_val - min_val == 0:
        logging.warning(f"Insulation scores are constant in column {column_name}. Normalization skipped.")
        return df[column_name] 
    
    return (df[column_name] - min_val) / (max_val - min_val)


def detect_tad_boundaries(insulation_scores, valid_mask, min_distance=5, prominence=0.1):
    logging.info("Detecting TAD boundaries...")

    try:
        working_scores = insulation_scores.copy()
        working_scores[~valid_mask] = 0 
    
        boundaries, properties = find_peaks(-working_scores, distance=min_distance, prominence=prominence)

        logging.info(f"Found {len(boundaries)} total boundaries with prominences: {properties['prominences']}")
        
        strong_boundaries = boundaries[properties['prominences'] > prominence]
        
        logging.info(f"Found {len(strong_boundaries)} strong TAD boundaries.")
        return strong_boundaries, properties

    except Exception as e:
        logging.error(f"Error detecting TAD boundaries: {str(e)}")
        return np.array([]), {}


def identify_tad_boundary_genes(df, min_distance=5, prominence=0.1):
    strong_boundary_genes = set()
    weak_boundary_genes = set()

    for chrom in df['Gene1_Chromosome'].unique():
        chrom_df = df[df['Gene1_Chromosome'] == chrom].copy() 
        
        #chrom_df["Gene1_Insulation_Score_Norm"] = normalize_insulation_scores(chrom_df, "Gene1_Insulation_Score")
        #chrom_df["Gene2_Insulation_Score_Norm"] = normalize_insulation_scores(chrom_df, "Gene2_Insulation_Score")

        #insulation_scores = chrom_df["Gene1_Insulation_Score_Norm"].values

        chrom_df["Combined_Insulation_Score"] = (chrom_df["Gene1_Insulation_Score"] + chrom_df["Gene2_Insulation_Score"]) / 2

        insulation_scores = chrom_df["Combined_Insulation_Score"].values
        #print(f"Insulation score values: {insulation_scores}")
        valid_mask = ~np.isnan(insulation_scores)
        
        print(f"Chromosome {chrom}: Min Insulation Score = {insulation_scores.min()}, Max = {insulation_scores.max()}")
        
        print(f"Chromosome {chrom}: Number of valid insulation scores = {np.sum(valid_mask)}")
        
        strong_boundaries, properties = detect_tad_boundaries(insulation_scores, valid_mask)

        for idx in strong_boundaries:
            if idx < len(chrom_df):
                strong_boundary_genes.add(chrom_df.iloc[idx]["Gene1_clean"])
        
        weak_boundary_genes.update(set(chrom_df["Gene1_clean"]) - strong_boundary_genes)  

    print(f"Total Strong Boundary Genes: {len(strong_boundary_genes)}")
    print(f"Total Weak Boundary Genes: {len(weak_boundary_genes)}")

    print("\nStrong Boundary Genes:")
    print(strong_boundary_genes)
    
    print("\nWeak Boundary Genes:")
    print(weak_boundary_genes)

    return strong_boundary_genes, weak_boundary_genes


def get_enrichr_results(gene_list, database):
    if not gene_list:
        print(f"Skipping {database} (No genes available)")
        return pd.DataFrame()

    genes_str = '\n'.join(gene_list)
    payload = {
        'list': (None, genes_str),
        'description': (None, 'Mouse TAD Boundary Gene Analysis')  # Use mouse in the description for enrichr (more reliable results!)
    }
    
    response = requests.post(f"{ENRICHR_URL}/addList", files=payload)
    if not response.ok:
        raise Exception("Error submitting gene list to Enrichr.")
    
    data = json.loads(response.text)
    user_list_id = data['userListId']
    
    response = requests.get(f"{ENRICHR_URL}/enrich",
                            params={'userListId': user_list_id, 'backgroundType': database})
    if not response.ok:
        raise Exception("Error retrieving enrichment results from Enrichr.")
    
    return pd.DataFrame(json.loads(response.text)[database], 
                        columns=['Rank', 'Term', 'P-value', 'Z-score', 'Combined score', 
                                 'Overlapping genes', 'Adjusted p-value', 'Old p-value', 
                                 'Old adjusted p-value'])

def analyze_tad_boundaries_with_go(strong_boundaries, weak_boundaries, databases):
    if not os.path.exists("GO_results_TAD"):
        os.makedirs("GO_results_TAD")
    
    results = {}
    
    boundary_types = {
        "Strong_Boundaries": strong_boundaries,
        "Weak_Boundaries": weak_boundaries
    }

    for boundary_type, genes in boundary_types.items():
        print(f"\nRunning GO analysis for {boundary_type} ({len(genes)} genes)...")
        
        boundary_results = {}
        
        for db in databases:
            try:
                go_results = get_enrichr_results(genes, db)
                
                if not go_results.empty:
                    #file_name = f"GO_results_TAD/{boundary_type}_{db}.csv"
                    #go_results.to_csv(file_name, index=False)
                    print(f"Saved GO results for {boundary_type} - {db}")
                
                boundary_results[db] = go_results
            
            except Exception as e:
                print(f"Error analyzing {boundary_type} in {db}: {e}")
                boundary_results[db] = pd.DataFrame()  

        results[boundary_type] = boundary_results
    
    return results

def save_go_results_to_excel(results):
    with pd.ExcelWriter('GO_results_TAD/TAD_boundries_go_analyzes.xlsx') as writer:
        for boundary_type, db_results in results.items():
            for db, df in db_results.items():
                if not df.empty:
                    sheet_name = f"{boundary_type}_{db[:10]}" 
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print("\nGO results saved to 'GO_results_TAD/TAD_boundries_go_analyzes.xlsx'")

def run_tad_boundary_go_analysis(csv_file, min_distance=5, prominence=0.1):
    print("\nLoading time-series gene expression data...")
    
    df = pd.read_csv(csv_file)
    df['Gene1_clean'] = df['Gene1'].apply(clean_gene_name)
    df['Gene2_clean'] = df['Gene2'].apply(clean_gene_name)

    strong_boundaries, weak_boundaries = identify_tad_boundary_genes(df, min_distance=min_distance, 
                                                                     prominence=prominence)

    databases = ["GO_Biological_Process_2021", "GO_Molecular_Function_2021"]
    go_results = analyze_tad_boundaries_with_go(strong_boundaries, weak_boundaries, databases)

    save_go_results_to_excel(go_results)

    print("\nTAD Boundary GO Enrichment Analysis Completed.")

def plot_insulation_scores_with_boundaries(chrom_df, strong_boundaries, weak_boundaries, chromosome_name):
    plt.figure(figsize=(12, 6))
    plt.plot(chrom_df['Gene1_Insulation_Score'], label='Gene1 Insulation Score', color='b', alpha=0.6)
    plt.plot(chrom_df['Gene2_Insulation_Score'], label='Gene2 Insulation Score', color='g', alpha=0.6)

    strong_indices = chrom_df[chrom_df['Gene1_clean'].isin(strong_boundaries)].index.tolist()
    weak_indices = chrom_df[chrom_df['Gene1_clean'].isin(weak_boundaries)].index.tolist()

    plt.scatter(strong_indices, chrom_df.loc[strong_indices, 'Gene1_Insulation_Score'], 
                color='red', label='Strong Boundaries', zorder=5)
    plt.scatter(weak_indices, chrom_df.loc[weak_indices, 'Gene1_Insulation_Score'], 
                color='orange', label='Weak Boundaries', zorder=5)
    
    plt.title(f"Insulation Scores and TAD Boundaries for Chromosome {chromosome_name}")
    plt.xlabel("Gene Index") # ordering of genes 
    plt.ylabel("Insulation Score")
    plt.legend()
    plt.grid(True)

    plt.savefig(f'GO_results_TAD/insulation_score_with_boundries_{chromosome_name}.pdf')
    plt.show()


if __name__ == "__main__":
    csv_file = "/Users/beyzakaya/Desktop/temporal gene/mapped/enhanced_interactions_synthetic_simple.csv"

    df = pd.read_csv(csv_file)
    df['Gene1_clean'] = df['Gene1'].apply(clean_gene_name)
    df['Gene2_clean'] = df['Gene2'].apply(clean_gene_name)
    
    #for chrom in df['Gene1_Chromosome'].unique():
    #    chrom_df = df[df['Gene1_Chromosome'] == chrom]
    #    plot_insulation_scores_with_boundaries(chrom_df, strong_boundaries, weak_boundaries, chrom)

    all_chromosomes = set(df['Gene1_Chromosome'].unique()).union(set(df['Gene2_Chromosome'].unique()))
    for chrom in all_chromosomes:
        chrom_df = df[(df['Gene1_Chromosome'] == chrom) | (df['Gene2_Chromosome'] == chrom)]
        strong_boundaries, weak_boundaries = identify_tad_boundary_genes(chrom_df)
        plot_insulation_scores_with_boundaries(chrom_df, strong_boundaries, weak_boundaries, chrom)
   
    #run_tad_boundary_go_analysis(csv_file, min_distance=5, prominence=0.1)
