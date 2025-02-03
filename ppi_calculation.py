import requests
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os 
from create_graph_and_embeddings_STGCN import clean_gene_name

def get_mouse_ppi_data(dataset):
    #clean_genes = [clean_gene_name(gene) for gene in gene_list]
    clean_genes = dataset.node_map.keys()
    string_api_url = "https://string-db.org/api/json/network"
    params = {
        "identifiers": "\r".join(clean_genes),
        "species": 10090,  # Mouse
        "required_score": 400
    }
    
    try:
        response = requests.post(string_api_url, data=params)
        ppi_data = response.json()
        return ppi_data
    except Exception as e:
        print(f"Error fetching PPI data: {e}")
        return None

def compare_ppi_with_hic(dataset, ppi_data):
    genes = set(dataset.df['Gene1'].unique()) | set(dataset.df['Gene2'].unique())
    clean_genes = dataset.node_map.keys()
    
    ppi_network = nx.Graph()
    for interaction in ppi_data:
        gene1, gene2 = interaction['preferredName_A'], interaction['preferredName_B']
        if gene1 in clean_genes and gene2 in clean_genes:
            ppi_network.add_edge(gene1, gene2, weight=float(interaction['score']))
    
    hic_edges = set()
    for _, row in dataset.df.iterrows():
        hic_edges.add((row['Gene1'], row['Gene2']))
    
    overlapping = set(ppi_network.edges()) & hic_edges
    
    return {
        'ppi_edges': len(ppi_network.edges()),
        'hic_edges': len(hic_edges),
        'overlapping': len(overlapping)
    }

def get_mgi_info(dataset):
    gene_list = dataset.node_map.keys()

    mgi_base_url = "http://www.informatics.jax.org/marker"
    gene_info = {}
    
    for gene in gene_list:
        try:
            response = requests.get(f"{mgi_base_url}/{gene}")
            if response.status_code == 200:
                gene_info[gene] = response.text
        except Exception as e:
            print(f"Error fetching MGI data for {gene}: {e}")
    
    return gene_info

def analyze_and_plot_ppi(dataset, target_genes=None, save_path='plottings_STGCN/ppi_network.png'):
    genes = target_genes if target_genes else list(set(dataset.df['Gene1_clean'].unique()) | 
                                                 set(dataset.df['Gene2_clean'].unique()))    
    params = {
        "identifiers": "\r".join(genes),
        "species": 10090,
        "required_score": 200
    }
    
    try:
        response = requests.post("https://string-db.org/api/json/network", data=params)
        ppi_data = response.json()
        ppi_df = pd.DataFrame(ppi_data)
        
        G_ppi = nx.Graph()
        for _, row in ppi_df.iterrows():
            gene1 = clean_gene_name(row['preferredName_A'])
            gene2 = clean_gene_name(row['preferredName_B'])
            G_ppi.add_edge(gene1, gene2, weight=float(row['score']))
        
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G_ppi)
        nx.draw(G_ppi, pos, with_labels=True, node_size=1000)
        plt.title("PPI Network")
        plt.savefig(save_path)
        plt.close()
        hic_interactions = {(clean_gene_name(row['Gene1']), clean_gene_name(row['Gene2'])) 
                          for _, row in dataset.df.iterrows()}
        
        overlapping = set(G_ppi.edges()) & hic_interactions
        
        print(f"\nPPI Analysis Results:")
        print(f"Total PPI interactions: {len(G_ppi.edges())}")
        print(f"Total HiC interactions: {len(hic_interactions)}")
        print(f"Overlapping interactions: {len(overlapping)}")
        
        return ppi_df, G_ppi
        
    except Exception as e:
        print(f"Error in PPI analysis: {e}")
        return None, None