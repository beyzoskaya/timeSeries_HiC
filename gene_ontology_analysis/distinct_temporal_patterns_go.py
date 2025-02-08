import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.cluster import KMeans
import pandas as pd
import requests
import json
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

ENRICHR_URL = "https://maayanlab.cloud/Enrichr"

def clean_gene_name(gene_name):
    if pd.isna(gene_name):
        return gene_name
    return gene_name.split('(')[0].strip()

def extract_expression_values(df, time_points):
    all_expressions = []
    expression_values = {}

    for gene in set(df['Gene1_clean']).union(set(df['Gene2_clean'])):
        gene_expressions = []
        for t in time_points:
            gene1_expr = df[df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
            gene2_expr = df[df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
            expr_value = gene1_expr[0] if len(gene1_expr) > 0 else \
                        (gene2_expr[0] if len(gene2_expr) > 0 else 0.0)
            gene_expressions.append(expr_value)
        
        expression_values[gene] = gene_expressions
        all_expressions.extend(gene_expressions)

    global_min, global_max = min(all_expressions), max(all_expressions) 
    for gene in expression_values:
        expression_values[gene] = [(x - global_min) / (global_max - global_min) for x in expression_values[gene]] # normalize expression values same as in embedding creation

    return expression_values

def identify_temporal_clusters(expression_values, n_clusters=3):
    genes = list(expression_values.keys())
    expression_matrix = np.array([expression_values[gene] for gene in genes])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(expression_matrix)

    clusters = {i: [] for i in range(n_clusters)}
    cluster_trends = {i: [] for i in range(n_clusters)}

    for gene, label in zip(genes, cluster_labels):
        clusters[label].append(gene)
        cluster_trends[label].append(expression_values[gene])

    cluster_means = {i: np.mean(cluster_trends[i], axis=0) for i in range(n_clusters)}

    cluster_types = {}
    for cluster_id, trend in cluster_means.items():
        slope = np.polyfit(range(len(trend)), trend, 1)[0]  # Calculate slope over time points
        print(f"Slope: {slope}")
        if slope > 0.001:  
            cluster_types[cluster_id] = "Upregulated"
        elif slope < 0:
            cluster_types[cluster_id] = "Downregulated"
        else:
            cluster_types[cluster_id] = "Stable"

    for cluster_id, cluster_type in cluster_types.items():
        print(f"Cluster {cluster_id} ({cluster_type}): {len(clusters[cluster_id])} genes")
    
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(expression_matrix)

    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels)
    plt.title("PCA of Gene Expression Clusters")
    #plt.show()
    plt.savefig("GO_results_temporal_patterns/pca_gene_clusters.png")

    return clusters, cluster_types


def get_enrichr_results(gene_list, database):
    genes_str = '\n'.join(gene_list)
    payload = {
        'list': (None, genes_str),
        'description': (None, 'Temporal Gene Cluster Analysis')
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

def analyze_clusters_with_go(clusters, databases=["GO_Biological_Process_2021", "GO_Molecular_Function_2021"]):
    if not os.path.exists("GO_results_temporal_patterns"):
        os.makedirs("GO_results_temporal_patterns")
    
    results = {}

    excel_filename = "temporal_go_analysis.xlsx"
    excel_file_pth = f"GO_results_temporal_patterns/{excel_filename}"
    with pd.ExcelWriter(excel_file_pth) as writer:
        for cluster_id, gene_list in clusters.items():
            print(f"\nRunning GO analysis for Cluster {cluster_id} ({len(gene_list)} genes)...")
            cluster_results = {}

            for db in databases:
                try:
                    go_results = get_enrichr_results(gene_list, db)
                    
                    if not go_results.empty:
                        sheet_name = f"Cluster_{cluster_id}_{db.split('_')[1]}"  # Shorter sheet name
                        go_results.to_excel(writer, sheet_name=sheet_name, index=False)
                        print(f"Saved GO results for Cluster {cluster_id} - {db} to Excel")

                    cluster_results[db] = go_results
                
                except Exception as e:
                    print(f"Error analyzing Cluster {cluster_id} in {db}: {e}")
                    cluster_results[db] = pd.DataFrame()  
            
            results[cluster_id] = cluster_results
    
    print(f"\nGO results saved in '{excel_file_pth}'")
    return results

def run_temporal_clustering_and_go_analysis(expression_data_csv, num_clusters=3, expression_values=None):
    if expression_values is None:
        raise ValueError("Expression values must be provided!")

    print("\nPerforming Temporal Clustering...")
    clusters, cluster_types = identify_temporal_clusters(expression_values, num_clusters)
    print(f"Cluster types: {cluster_types}")

    for cluster, genes in clusters.items():
        print(f"\nCluster {cluster} : {len(genes)} genes")
        #print(", ".join(genes[:10]) + ("..." if len(genes) > 10 else ""))
        print(", ".join(genes)) 

    print("\n🔍 Running GO Enrichment Analysis...")
    go_results = analyze_clusters_with_go(clusters)

    print("\nTemporal clustering & GO enrichment analysis completed!")
    print("Results saved in 'temporal_go_analysis.xlsx'")

if __name__ == "__main__":

    csv_file = "/Users/beyzakaya/Desktop/temporal gene/mapped/enhanced_interactions_synthetic_simple.csv"
    df = pd.read_csv(csv_file)
    df['Gene1_clean'] = df['Gene1'].apply(clean_gene_name)
    df['Gene2_clean'] = df['Gene2'].apply(clean_gene_name)

    time_points = [float(col.split('_')[-1]) for col in df.columns if "Gene1_Time" in col]
    expression_values = extract_expression_values(df, time_points)

    run_temporal_clustering_and_go_analysis(csv_file, num_clusters=3, expression_values=expression_values)

    
