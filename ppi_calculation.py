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
        "required_score": 400
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

def analyze_ppi_with_aliases(dataset, save_path='plottings_STGCN/ppi_network.png'):
   
   gene_mappings = {
   'TTF-1': ['Nkx2-1', 'Titf1', 'T/EBP'],
   'AMACR': ['Amacr', 'RCDP2'],
   'ABCG2': ['Bcrp1', 'Bcrp', 'CD338'],
   'MMP7': ['Mmp7', 'Matrilysin'],
   'HPGDS': ['Pgds2', 'H-pgds'],
   'VIM': ['Vim'],
   'RAGE': ['Ager'],
   'TGFB1': ['Tgfb1'],
   'EGFR': ['Egfr', 'ErbB1'],
   'FOXF2': ['Foxf2'],
   'Hist1h1b': ['H1-5', 'H1f5'],
   'P-63': ['Trp63', 'p63'],
   'INMT': ['Inmt'],
   'ADAMTSL2': ['Adamtsl2'],
   'Tnc': ['TN-C'],
   'FGF18': ['Fgf18'],
   'CD38': ['Cd38'],
   'MMP-3': ['Mmp3', 'Stromelysin-1'],
   'Lrp2': ['Megalin'],
   'PPIA': ['Ppia', 'CypA'],
   'THTPA': ['Thtpa'],
   'VEGF': ['Vegfa', 'Vgf'],
   'GATA-6': ['Gata6'],
   'ABCA3': ['Abc3'],
   'TFRC': ['Tfrc', 'CD71'],
   'F13A1': ['F13a1'],
   'KCNMA1': ['Kcnma1', 'Slo1'],
   'EPHA7': ['Epha7'],
   'HMBS': ['Hmbs', 'Pbgd'],
   'E2F8': ['E2f8'],
   'Claudin5': ['Cldn5'],
   'GUCY1A2': ['Gucy1a2'],
   'PRIM2': ['Prim2'],
   'TBP': ['Tbp', 'TATA-BP'],
   'N-Cadherin': ['Cdh2'],
   'Thy1': ['CD90'],
   'Claudin1': ['Cldn1'],
   'IGFBP3': ['Igfbp3'],
   'YWHAZ': ['Ywhaz', '14-3-3-zeta'],
   'HPRT': ['Hprt1'],
   'ABCD1': ['Abcd1'],
   'NME3': ['Nme3'],
   'MGAT4A': ['Mgat4a'],
   'MCPt4': ['Mcpt4'],
   'SFTP-D': ['Sftpd', 'SP-D'],
   'Shisa3': ['Shisa3'],
   'integrin subunit alpha 8': ['Itga8']
}
   
   genes = list(set(dataset.df['Gene1_clean'].unique()) | set(dataset.df['Gene2_clean'].unique()))
   all_identifiers = []
   gene_to_alias = {}
   
   for gene in genes:
       if gene in gene_mappings:
           all_identifiers.extend(gene_mappings[gene])
           for alias in gene_mappings[gene]:
               gene_to_alias[alias] = gene
       else:
           all_identifiers.append(gene)
           gene_to_alias[gene] = gene
   
   params = {
       "identifiers": "\r".join(all_identifiers),
       "species": 10090,
       "required_score": 400
   }
   
   try:
       response = requests.post("https://string-db.org/api/json/network", data=params)
       ppi_data = response.json()
       ppi_df = pd.DataFrame(ppi_data)
       
       G_ppi = nx.Graph()
       for _, row in ppi_df.iterrows():
           gene1 = gene_to_alias.get(row['preferredName_A'], row['preferredName_A'])
           gene2 = gene_to_alias.get(row['preferredName_B'], row['preferredName_B'])
           G_ppi.add_edge(gene1, gene2, weight=float(row['score']))
       
       plt.figure(figsize=(12, 12))
       pos = nx.spring_layout(G_ppi)
       nx.draw(G_ppi, pos, with_labels=True, node_size=1000)
       plt.title("PPI Network (with Alias Mapping)")
       plt.savefig(save_path)
       plt.close()
       
       # Compare with HiC
       hic_interactions = {(row['Gene1_clean'], row['Gene2_clean']) for _, row in dataset.df.iterrows()}
       overlapping = set(G_ppi.edges()) & hic_interactions
       
       print(f"\nPPI Analysis Results:")
       print(f"Total PPI interactions: {len(G_ppi.edges())}")
       print(f"Total HiC interactions: {len(hic_interactions)}")
       print(f"Overlapping interactions: {len(overlapping)}")
       
       return ppi_df, G_ppi
       
   except Exception as e:
       print(f"Error in PPI analysis: {e}")
       return None, None