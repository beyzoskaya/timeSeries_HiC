import pandas as pd 
import os
import json
import requests
from distinct_temporal_patterns_go import clean_gene_name
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import wrap

ENRICHR_URL = "https://maayanlab.cloud/Enrichr"

def get_enrichr_results(gene_list, database):
    genes_str = '\n'.join(gene_list)
    description = 'Mouse gene compartment analysis' 
    payload = {
        'list': (None, genes_str),
        'description': (None, description)
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

def analyze_compartments_with_go(df, databases=["GO_Biological_Process_2021", "GO_Molecular_Function_2021"]):
    if not os.path.exists("GO_results_compartments"):
        os.makedirs("GO_results_compartments")
    
    all_genes = pd.concat([df['Gene1_clean'], df['Gene2_clean']]).unique()

    compartment_A_genes = df[df['Gene1_Compartment'] == 'A']['Gene1_clean'].unique()
    compartment_B_genes = df[df['Gene1_Compartment'] == 'B']['Gene1_clean'].unique()

    remaining_genes = np.setdiff1d(all_genes, np.concatenate([compartment_A_genes, compartment_B_genes]))

    #missing_compartments = df[df['Gene1_Compartment'].isna()]
    #print(f"Genes with missing compartment information: {missing_compartments}")

    #mismatched_compartments = df[df['Gene1_Compartment'] != df['Gene2_Compartment']]
    #print(f"Genes with mismatched compartment info: {mismatched_compartments}")

    #duplicated_genes = df[df.duplicated(subset=['Gene1_clean', 'Gene2_clean'], keep=False)]
    #print(f"Duplicated genes: {duplicated_genes}")

    results = {}

    excel_filename = "compartment_go_analysis.xlsx"
    excel_file_pth = f"GO_results_compartments/{excel_filename}"
    with pd.ExcelWriter(excel_file_pth) as writer:
        print(f"\nRunning GO analysis for Compartment A ({len(compartment_A_genes)} genes)...")
        compartment_A_results = {}
        for db in databases:
            try:
                go_results_A = get_enrichr_results(compartment_A_genes, db)
                
                if not go_results_A.empty:
                    sheet_name = f"Compartment_A_{db.split('_')[1]}"
                    go_results_A.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"Saved GO results for Compartment A - {db} to Excel")

                compartment_A_results[db] = go_results_A
            
            except Exception as e:
                print(f"Error analyzing Compartment A in {db}: {e}")
                compartment_A_results[db] = pd.DataFrame()

        print(f"\nRunning GO analysis for Compartment B ({len(compartment_B_genes)} genes)...")
        compartment_B_results = {}
        for db in databases:
            try:
                go_results_B = get_enrichr_results(compartment_B_genes, db)
                
                if not go_results_B.empty:
                    sheet_name = f"Compartment_B_{db.split('_')[1]}"
                    go_results_B.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"Saved GO results for Compartment B - {db} to Excel")

                compartment_B_results[db] = go_results_B
            
            except Exception as e:
                print(f"Error analyzing Compartment B in {db}: {e}")
                compartment_B_results[db] = pd.DataFrame()
        
        print(f"\nRunning GO analysis for Compartment A and B ({len(remaining_genes)} genes)...")
        compartment_both_results = {}
        for db in databases:
            try:
                go_results_both = get_enrichr_results(remaining_genes, db)
                
                if not go_results_both.empty:
                    sheet_name = f"Compartment_AB_{db.split('_')[1]}"
                    go_results_both.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"Saved GO results for Compartment A and B - {db} to Excel")

                compartment_both_results[db] = compartment_both_results
            
            except Exception as e:
                print(f"Error analyzing Compartment A and B (both) in {db}: {e}")
                compartment_both_results[db] = pd.DataFrame()
        
        results['Compartment_A'] = compartment_A_results
        results['Compartment_B'] = compartment_B_results
        results['Compartment_AB'] = compartment_both_results
    
    print(f"\nGO results saved in '{excel_file_pth}'")
    return results

def run_compartment_go_analysis(csv_file):
    df = pd.read_csv(csv_file)
    df['Gene1_clean'] = df['Gene1'].apply(clean_gene_name)
    df['Gene2_clean'] = df['Gene2'].apply(clean_gene_name)
    print(f"Length of cleaned gene names: {len(df['Gene2_clean'])}")

    print("\nRunning GO Enrichment Analysis for Compartments...")
    go_results = analyze_compartments_with_go(df)

    print("\nCompartment-based GO enrichment analysis completed!")
    print("Results saved in 'compartment_go_analysis.xlsx'")
    
def read_go_data(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df

def plot_go_terms(df, compartment_name):
    """
    Top GO terms for each compartment
    """
    df_sorted = df.sort_values(by='Combined score', ascending=False).head(10)
    plt.figure(figsize=(40, 12))
    sns.barplot(y=df_sorted['Term'], x=df_sorted['Combined score'], palette='viridis')
    plt.xlabel("Combined Score")
    plt.ylabel("GO Term")
    plt.title(f"Top Molecular Function GO Terms - {compartment_name}")
    plt.gca().invert_yaxis()
    #plt.show()
    plt.savefig(f"GO_results_compartments/molecular_func_terms_{compartment_name}.pdf")

def compare_compartments(df_A, df_B):
    df_A['Compartment'] = 'A'
    df_B['Compartment'] = 'B'
    df_combined = pd.concat([df_A, df_B])
    df_top = df_combined.sort_values(by='Combined score', ascending=False).head(10)
    
    plt.figure(figsize=(36, 12))
    sns.barplot(x='Combined score', y='Term', hue='Compartment', data=df_top, palette=['blue', 'red'])
    plt.xlabel("Combined Score")
    plt.ylabel("GO Term")
    plt.title("Comparison of GO Terms in Compartments A and B")
    #plt.show()
    plt.savefig("GO_results_compartments/go_terms_comparison_for_compartments.pdf")

def plot_go_heatmap(df_a, df_b):
    """
    Identifies which GO terms are more enriched in which compartment
    Highlights the strongest biological functions with color intensity
    """
    df_a = df_a[['Term', 'Combined score']].rename(columns={'Combined score': 'Compartment A'})
    df_b = df_b[['Term', 'Combined score']].rename(columns={'Combined score': 'Compartment B'})
    
    df_merged = pd.merge(df_a, df_b, on="Term", how="outer").fillna(0)

    df_merged['Max Score'] = df_merged[['Compartment A', 'Compartment B']].max(axis=1)
    df_merged = df_merged.sort_values(by='Max Score', ascending=False).head(20)  
    df_merged = df_merged.drop(columns=['Max Score']).set_index("Term")

    plt.figure(figsize=(40, 12))
    sns.heatmap(df_merged, cmap="viridis", annot=True, fmt=".2f", linewidths=0.5)
    plt.xlabel("Compartment")
    plt.ylabel("GO Terms")
    plt.title("Heatmap of GO Term Enrichment in Compartments A & B")

    plt.savefig("GO_results_compartments/go_term_heatmap.pdf")
    #plt.show()

def wrap_labels(labels, max_length=30):
    wrapped_labels = []
    for label in labels:
        wrapped_labels.append('\n'.join(wrap(label, max_length)))
    return wrapped_labels

def plot_go_bubble(df, compartment_name):
    """ 
    Horizontal Bubble plot of GO terms showing Combined Score, Z-score, and p-value
    For example, 
    If "ABC-type xenobiotic transporter activity" is bright yellow in Compartment A 
    but dark in B, it suggests that this function is more critical in A
    """
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

    plt.savefig(f"GO_results_compartments/go_term_bubble_{compartment_name}.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":

    #csv_file = "/Users/beyzakaya/Desktop/temporal gene/mapped/enhanced_interactions_synthetic_simple.csv"
    #run_compartment_go_analysis(csv_file)

    file_path = '/Users/beyzakaya/Desktop/temporal gene/gene_ontology_analysis/GO_results_compartments/compartment_go_analysis.xlsx'
    df_A = read_go_data(file_path, 'Compartment_A_Molecular')
    df_B = read_go_data(file_path, 'Compartment_B_Molecular')
    #plot_go_terms(df_A, "Compartment A")
    #plot_go_terms(df_B, "Compartment B")
    #compare_compartments(df_A, df_B)
    #plot_go_heatmap(df_A, df_B)

    plot_go_bubble(df_A, "Compartment_A")
    plot_go_bubble(df_B, "Compartment_B")

