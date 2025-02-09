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
import seaborn as sns
import textwrap

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
        
        strong_boundaries, _ = detect_tad_boundaries(insulation_scores, valid_mask)

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

def plot_combined_insulation_scores(chrom_df, strong_boundaries, weak_boundaries, chromosome_name):
    chrom_df["Combined_Insulation_Score"] = (chrom_df["Gene1_Insulation_Score"] + chrom_df["Gene2_Insulation_Score"]) / 2

    plt.figure(figsize=(12, 6))
    plt.plot(chrom_df['Combined_Insulation_Score'], label='Combined Insulation Score', color='purple', alpha=0.8)

    # boundaries to list for proper indexing
    strong_boundaries = list(strong_boundaries)
    weak_boundaries = list(weak_boundaries)

    # loc[] instead of iloc to access the gene names correctly
    plt.scatter(chrom_df.loc[chrom_df['Gene1_clean'].isin(strong_boundaries)].index, 
                chrom_df.loc[chrom_df['Gene1_clean'].isin(strong_boundaries), 'Combined_Insulation_Score'], 
                color='red', label='Strong Boundaries', zorder=5)
    
    plt.scatter(chrom_df.loc[chrom_df['Gene1_clean'].isin(weak_boundaries)].index, 
                chrom_df.loc[chrom_df['Gene1_clean'].isin(weak_boundaries), 'Combined_Insulation_Score'], 
                color='orange', label='Weak Boundaries', zorder=5)

    plt.title(f"Combined Insulation Scores and TAD Boundaries for Chromosome {chromosome_name}")
    plt.xlabel("Gene Index")
    plt.ylabel("Combined Insulation Score")
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f'GO_results_TAD/combined_insulation_score_with_boundries_{chromosome_name}.pdf')
    plt.show()


def plot_go_bubble(excel_file, sheet_name, compartment_name):
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    if all(col in df.columns for col in ['Term', 'Combined score', 'Z-score', 'P-value']):
    
        df_sorted = df.sort_values(by='Combined score', ascending=False).head(20)
        
        df_sorted['Term'] = wrap_labels(df_sorted['Term'], max_length=30)

        plt.figure(figsize=(24, 12))
        scatter = plt.scatter(
            x=df_sorted["Z-score"], 
            y=df_sorted["Term"], 
            s=df_sorted["Combined score"] * 2,  # Bubble size
            c=df_sorted["P-value"],  # Color represents p-value
            cmap="viridis", alpha=0.8, edgecolors="black"
        )

        plt.colorbar(scatter, label="P-value")
        plt.xlabel("Z-score (Enrichment Strength)")
        plt.ylabel("GO Term")
        plt.yticks(fontsize=10) 
        plt.title(f"GO Term Enrichment Bubble Plot - {compartment_name}")

        plt.savefig(f"GO_results_TAD/go_term_bubble_{compartment_name}.pdf", bbox_inches="tight")
        plt.show()
        
    else:
        print(f"Missing necessary columns in {sheet_name} for plotting.")

def wrap_labels(labels, max_width=30):
    return ['\n'.join(textwrap.wrap(label, max_width)) for label in labels]

def compare_tad_go_results(excel_file, tad_sheet_name, top_n=10):
    
    tad_results_df = pd.read_excel(excel_file, sheet_name=tad_sheet_name)
    
    if all(col in tad_results_df.columns for col in ['Term', 'P-value']):
        tad_results_sorted = tad_results_df.sort_values(by='P-value').head(top_n)
    
        tad_results_sorted['Wrapped_Term'] = wrap_labels(tad_results_sorted['Term'], max_width=30)

        plt.figure(figsize=(12, 14))
        plt.barh(tad_results_sorted['Wrapped_Term'], -np.log10(tad_results_sorted['P-value']), color='blue', alpha=0.6)

        plt.xlabel("-log10(p-value)")
        plt.ylabel("GO Terms")
        plt.title("Top GO Terms Enriched in TAD Boundaries")
        plt.gca().invert_yaxis()
        plt.savefig(f'GO_results_TAD/go_results_tad_{tad_sheet_name}.png', bbox_inches='tight')
        plt.show()
    
    else:
        print(f"Missing necessary columns in {tad_sheet_name} for plotting.")

def plot_tad_go_heatmap(excel_file, tad_sheet_name, top_n=10):
    tad_results_df = pd.read_excel(excel_file, sheet_name=tad_sheet_name)
    
    if all(col in tad_results_df.columns for col in ['Term', 'P-value']):
        tad_top_terms = tad_results_df.sort_values(by='P-value').head(top_n)

        heatmap_data = tad_top_terms[['Term', 'P-value']].set_index('Term')
        heatmap_data['-log10(P-value)'] = -np.log10(heatmap_data['P-value'])

        heatmap_data.index = wrap_labels(heatmap_data.index, max_width=30)

        plt.figure(figsize=(12, 14))
        sns.heatmap(heatmap_data[['-log10(P-value)']], annot=True, cmap='YlGnBu', cbar_kws={'label': '-log10(p-value)'})
        plt.title("Heatmap of GO Term Enrichment in TAD Boundaries")
        plt.xticks(rotation=45, ha="right") 
        plt.yticks(rotation=0)  
        plt.savefig(f'GO_results_TAD/go_results_heatmap_tad_{tad_sheet_name}.png', bbox_inches='tight')
        plt.show()
    
    else:
        print(f"Missing necessary columns in {tad_sheet_name} for heatmap plotting.")

if __name__ == "__main__":
    csv_file = "/Users/beyzakaya/Desktop/temporal gene/mapped/enhanced_interactions_synthetic_simple.csv"

    df = pd.read_csv(csv_file)
    df['Gene1_clean'] = df['Gene1'].apply(clean_gene_name)
    df['Gene2_clean'] = df['Gene2'].apply(clean_gene_name)
    
    strong_boundaries, weak_boundaries = identify_tad_boundary_genes(df, min_distance=5, prominence=0.1)

    print(f"Strong Boundary Genes: {strong_boundaries}")
    print(f"Weak Boundary Genes: {weak_boundaries}")
    
    #databases = ["GO_Biological_Process_2021", "GO_Molecular_Function_2021"]
    #go_results = analyze_tad_boundaries_with_go(strong_boundaries, weak_boundaries, databases)

    for chrom in df['Gene1_Chromosome'].unique():
        chrom_df = df[df['Gene1_Chromosome'] == chrom]
        #plot_insulation_scores_with_boundaries(chrom_df, strong_boundaries, weak_boundaries, chrom)
    
    for chrom in df['Gene1_Chromosome'].unique():
        chrom_df = df[df['Gene1_Chromosome'] == chrom]
        #plot_combined_insulation_scores(chrom_df, strong_boundaries, weak_boundaries, chrom)

    excel_file = "/Users/beyzakaya/Desktop/temporal gene/gene_ontology_analysis/GO_results_TAD/TAD_boundries_go_analyzes.xlsx" 
    sheet_name_strong_boundaries = 'Strong_Boundaries_GO_Molecul'
    sheet_name_weak_boundaries = 'Weak_Boundaries_GO_Molecul'
    compartment_name = "Compartment_B" # does not matter in the case of TAD boundary go analyzes/I tried to plot with different compartments 
    #plot_go_bubble(excel_file, sheet_name_weak_boundaries, compartment_name)

    compare_tad_go_results(excel_file, sheet_name_strong_boundaries, top_n=10)
    plot_tad_go_heatmap(excel_file, sheet_name_strong_boundaries, top_n=10)

    compare_tad_go_results(excel_file, sheet_name_weak_boundaries, top_n=10)
    plot_tad_go_heatmap(excel_file, sheet_name_weak_boundaries, top_n=10)

    print("TAD Boundary GO Enrichment Analysis Completed.")
   
    #run_tad_boundary_go_analysis(csv_file, min_distance=5, prominence=0.1)
