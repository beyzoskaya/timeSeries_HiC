import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
import matplotlib.pyplot as plt
import os
import seaborn as sns
from node2vec import Node2Vec
from scipy.stats import pearsonr
from STGCN.model.models import *
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
sys.path.append('./STGCN')
from STGCN.model.models import STGCNChebGraphConv
import argparse
from scipy.spatial.distance import cdist
from clustering_by_expr_levels import analyze_expression_levels_research, analyze_expression_levels_kmeans
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from networkx.algorithms.components import is_connected
from scipy import stats
from scipy.spatial.distance import pdist, squareform

def clean_gene_name(gene_name):
    """Clean gene name by removing descriptions and extra information"""
    if pd.isna(gene_name):
        return gene_name
    return gene_name.split('(')[0].strip()

def normalize_hic_weights(hic_values):
    # small constant to avoid log(0)
    eps = 1e-6
    
    # Log transformation to handle large variations
    log_weights = np.log1p(hic_values + eps)
    
    # Min-max scaling to [0,1]
    normalized = (log_weights - np.min(log_weights)) / (np.max(log_weights) - np.min(log_weights) + eps)
    
    return normalized

class TemporalGraphDataset:
    def __init__(self, csv_file, embedding_dim=64, seq_len=5, pred_len=1, graph_params=None, node2vec_params=None): # I change the seq_len to more lower value
        #self.graph_params = graph_params or {}
        #self.node2vec_params = node2vec_params or {}
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.embedding_dim = embedding_dim
        
        self.df = pd.read_csv(csv_file)
        print(f"Before cleaning the gene names Gene1: {self.df['Gene1']}")
        print(f"Before cleaning the gene names Gene2: {self.df['Gene2']}")
        #self.df['Gene1_clean'] = self.df['Gene1'].apply(clean_gene_name)
        #self.df['Gene2_clean'] = self.df['Gene2'].apply(clean_gene_name)

        #genes_to_remove = {'THTPA', 'AMACR', 'MMP7', 'ABCG2', 'HPGDS', 'VIM'}
        # Filter out genes which gave negative correlation to see performance of other genes 
        #self.df = self.df[~self.df['Gene1_clean'].isin(genes_to_remove)]
        #self.df = self.df[~self.df['Gene2_clean'].isin(genes_to_remove)]
        
        # Create static node mapping

        """
        I added this to not change proper parts for cleaning gene name because in miRNA data, all genes are in cleaned format
        """
        self.df['Gene1_clean'] = self.df['Gene1']
        self.df['Gene2_clean'] = self.df['Gene2']
        self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed', case=False)]

        unique_genes = pd.concat([self.df['Gene1_clean'], self.df['Gene2_clean']]).unique()
        self.node_map = {gene: idx for idx, gene in enumerate(unique_genes)}
        print(f"Unique genes: {self.node_map}")
        self.num_nodes = len(self.node_map)
        print(f"Number of nodes: {self.num_nodes}")
        
        # Get time points
        self.time_cols = [col for col in self.df.columns if 'Time_' in col]
        # This is for mRNA data because column names for time points are different
        self.time_points = sorted(list(set([float(col.split('_')[-1])  
                                          for col in self.time_cols if 'Gene1' in col])))
        self.time_points = [float(tp) for tp in self.time_points] # added for solving type error of time_points
      
        print(f"Found {len(self.time_points)} time points")
        print("Extracted time points:", self.time_points)
        
        # Create base graph and features
        self.base_graph = self.create_base_graph()
        print("Base graph created")
        #self.node_features = self.create_temporal_node_features_several_graphs_created_clustering() # try with several graphs for time series consistency
        self.node_features, self.temporal_edge_indices, self.temporal_edge_attrs = \
        self.create_temporal_node_features_several_graphs_created_mirna(debug_mode=True)
        print("Temporal node features created")
        
        # Get edge information
        self.edge_index, self.edge_attr = self.get_edge_index_and_attr()
        print(f"Graph structure created with {len(self.edge_attr)} edges")
    
    def create_base_graph(self):
        """Create a single base graph using structural features"""
        G = nx.Graph()
        G.add_nodes_from(self.node_map.keys())
        
        for _, row in self.df.iterrows():
            hic_weight = row['HiC_Interaction']
            compartment_sim = 1 if row['Gene1_Compartment'] == row['Gene2_Compartment'] else 0
            tad_dist = abs(row['Gene1_TAD_Boundary_Distance'] - row['Gene2_TAD_Boundary_Distance'])
            tad_sim = 1 / (1 + tad_dist)
            ins_sim = 1 / (1 + abs(row['Gene1_Insulation_Score'] - row['Gene2_Insulation_Score']))
            
            # Use only structural features for base graph
            weight = (hic_weight * 0.4 + 
                     compartment_sim * 0.2 + 
                     tad_sim * 0.2 + 
                     ins_sim * 0.2)
            
            G.add_edge(row['Gene1_clean'], row['Gene2_clean'], weight=weight)
        
        return G
    
    
    def create_temporal_node_features_several_graphs_created(self):
        temporal_features = {}
        
        # normalize all expression values across all time points
        print("\nNormalizing expression values across all time points...")
        all_expressions = []
        for t in self.time_points:
            for gene in self.node_map.keys():
                gene1_expr = self.df[self.df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
                gene2_expr = self.df[self.df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
                expr_value = gene1_expr[0] if len(gene1_expr) > 0 else \
                            (gene2_expr[0] if len(gene2_expr) > 0 else 0.0)
                all_expressions.append(expr_value)

                #print(f"\nTime point {t}:")
                #print(f"Expression range: [{np.min(all_expressions):.4f}, {np.max(all_expressions):.4f}]")
                #print(f"Mean: {np.mean(all_expressions):.4f}")
                #print(f"Std: {np.std(all_expressions):.4f}")
        
        # Global normalization parameters
        global_min = min(all_expressions)
        global_max = max(all_expressions)
        print(f"Global expression range: [{global_min:.4f}, {global_max:.4f}]")

        for t in self.time_points:
            print(f"\nProcessing time point {t}")
            
            expression_values = {}
            for gene in self.node_map.keys():
                gene1_expr = self.df[self.df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
                gene2_expr = self.df[self.df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
                expr_value = gene1_expr[0] if len(gene1_expr) > 0 else \
                            (gene2_expr[0] if len(gene2_expr) > 0 else 0.0)
                # min-max global normalization for mRNA
                expression_values[gene] = (expr_value - global_min) / (global_max - global_min)
            
            # Create graph using normalized expression values
            G = nx.Graph()
            G.add_nodes_from(self.node_map.keys())
            
            # edges with normalized weights
            for _, row in self.df.iterrows():
                gene1 = row['Gene1_clean']
                gene2 = row['Gene2_clean']
                
                # Use normalized expression values for similarity calculation
                expr_sim = 1 / (1 + abs(expression_values[gene1] - expression_values[gene2]))
                
                # Calculate other similarities
                hic_weight = row['HiC_Interaction']
                compartment_sim = 1 if row['Gene1_Compartment'] == row['Gene2_Compartment'] else 0
                tad_dist = abs(row['Gene1_TAD_Boundary_Distance'] - row['Gene2_TAD_Boundary_Distance'])
                tad_sim = 1 / (1 + tad_dist)
                ins_sim = 1 / (1 + abs(row['Gene1_Insulation_Score'] - row['Gene2_Insulation_Score']))
                
                # Combine weights
                weight = (hic_weight * 0.25 +
                        compartment_sim * 0.1 +
                        tad_sim * 0.1 +
                        ins_sim * 0.1 +
                        expr_sim * 0.45)
                
                G.add_edge(gene1, gene2, weight=weight)
            
            # Create Node2Vec embeddings
            node2vec = Node2Vec(
                G,
                dimensions=self.embedding_dim,
                walk_length=20,
                num_walks=100,
                p=1.0,
                q=1.0,
                workers=1,
                seed=42
            )
            
            model = node2vec.fit(
                window=15,
                min_count=1,
                batch_words=4
            )
            
            # Create feature vectors
            features = []
            for gene in self.node_map.keys():
                # Get Node2Vec embedding
                node_embedding = torch.tensor(model.wv[gene], dtype=torch.float32)
                
                # Normalize embedding
                node_embedding = (node_embedding - node_embedding.min()) / (node_embedding.max() - node_embedding.min() + 1e-8)
                
                # Last dimension is the normalized expression value
                node_embedding[-1] = expression_values[gene]
                
                features.append(node_embedding)
            
            temporal_features[t] = torch.stack(features)

            # After creating Node2Vec embeddings
            print("\n=== Node2Vec Embedding Statistics ===")
            print(f"Embedding shape: {temporal_features[t].shape}")
            print(f"Range: [{temporal_features[t].min():.4f}, {temporal_features[t].max():.4f}]")
            print(f"Mean: {temporal_features[t].mean():.4f}")
            print(f"Std: {temporal_features[t].std():.4f}")
            print("Sample embeddings for first 3 nodes:")
            for i in range(min(3, len(self.node_map))):
                gene = list(self.node_map.keys())[i]
                print(f"{gene}: {temporal_features[t][i, :5]}")
            
            # Print diagnostics
            print(f"\nFeature Statistics for time {t}:")
            print(f"Expression range: [{min(expression_values.values()):.4f}, {max(expression_values.values()):.4f}]")
            print(f"Embedding range: [{temporal_features[t].min():.4f}, {temporal_features[t].max():.4f}]")
            print(f"Sample embeddings for first 3 genes:")
            for i in range(3):
                gene = list(self.node_map.keys())[i]
                print(f"{gene}: {temporal_features[t][i, :5]}, Expression: {expression_values[gene]:.4f}")
        
        return temporal_features
    
    def create_temporal_node_features_several_graphs_created_clustering(self):
        temporal_features = {}
        temporal_edge_indices = {}
        temporal_edge_attrs = {}

        clusters, _ = analyze_expression_levels_research(self)
        #clusters, _ = analyze_expression_levels_kmeans(self)
        gene_clusters = {}
        for cluster_name, genes in clusters.items():
            for gene in genes:
                gene_clusters[gene] = cluster_name
        
        # normalize all expression values across all time points
        print("\nNormalizing expression values across all time points...")
        all_expressions = []
        high_expr = []
        low_expr = []
        for t in self.time_points:
            for gene in self.node_map.keys():
                gene1_expr = self.df[self.df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
                gene2_expr = self.df[self.df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
                expr_value = gene1_expr[0] if len(gene1_expr) > 0 else \
                            (gene2_expr[0] if len(gene2_expr) > 0 else 0.0)
                all_expressions.append(expr_value)

        # Global normalization parameters
        global_min = min(all_expressions)
        global_max = max(all_expressions)
        print(f"Global expression range: [{global_min:.4f}, {global_max:.4f}]")

        if global_max == global_min:
            print("Warning: Global expression range has zero range, which may cause division by zero.")

        low_corr_genes = ['AMACR', 'ABCG2', 'MMP7', 'HPGDS', 'MGAT4A']

        for t in self.time_points:
            print(f"\nProcessing time point {t}")
            #print(f"All time points: {self.time_points}")
            
            expression_values = {}
            for gene in self.node_map.keys():
                gene1_expr = self.df[self.df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
                gene2_expr = self.df[self.df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
                
                # Added for debugging of miRNA data
                if np.isnan(gene1_expr).any():
                    print(f"NaN detected in Gene1 expression for {gene} at time {t}")
                if np.isnan(gene2_expr).any():
                    print(f"NaN detected in Gene2 expression for {gene} at time {t}")

                expr_value = gene1_expr[0] if len(gene1_expr) > 0 else \
                            (gene2_expr[0] if len(gene2_expr) > 0 else 0.0)

                # min-max normalization for mRNA
                expression_values[gene] = (expr_value - global_min) / (global_max - global_min + 1e-8)


            # Create graph using normalized expression values and cluster information
            G = nx.Graph()
            G.add_nodes_from(self.node_map.keys())

            # Add edges with cluster-aware weights
            edge_index = []
            edge_weights = []
            for _, row in self.df.iterrows():
                #hic_weight = hic_mapping[idx]
                gene1 = row['Gene1_clean']
                gene2 = row['Gene2_clean']
                
                # Calculate edge weight components
                expr_sim = 1 / (1 + abs(expression_values[gene1] - expression_values[gene2]))

                # Added for debugging of miRNA data
                if np.isnan(expr_sim):
                    print(f"NaN detected in expression similarity between {gene1} and {gene2} at time {t}")

                if gene1 in low_corr_genes or gene2 in low_corr_genes:
                    hic_weight = np.log1p(row['HiC_Interaction'])  # Log transform for the low correlated genes
                else:
                    hic_weight = row['HiC_Interaction'] 
                
                if pd.isna(row['HiC_Interaction']):
                    print(f"HiC weight is NaN")
            
                compartment_sim = 1 if row['Gene1_Compartment'] == row['Gene2_Compartment'] else 0 

                tad_dist = abs(row['Gene1_TAD_Boundary_Distance'] - row['Gene2_TAD_Boundary_Distance'])
                tad_sim = 1 / (1 + tad_dist)
                ins_sim = 1 / (1 + abs(row['Gene1_Insulation_Score'] - row['Gene2_Insulation_Score']))

                # Added for debugging of miRNA data
                if np.isnan(hic_weight):
                    print(f"NaN detected in HiC weight for {gene1}-{gene2}")
                if np.isnan(compartment_sim):
                    print(f"NaN detected in compartment similarity for {gene1}-{gene2}")
                if np.isnan(tad_sim):
                    print(f"NaN detected in TAD similarity for {gene1}-{gene2}")
                if np.isnan(ins_sim):
                    print(f"NaN detected in insulation similarity for {gene1}-{gene2}")
                
                # cluster similarity component
                cluster_sim = 1.2 if gene_clusters[gene1] == gene_clusters[gene2] else 1.0

                # Artificially increase connection strength for negative correlated genes
                if gene1 in ["AMACR", "ABCG2", "HPGDS"] or gene2 in ["AMACR", "ABCG2", "HPGDS"]:
                    weight = (hic_weight * 0.2 +  
                            compartment_sim * 0.1 +
                            tad_sim * 0.1 +
                            ins_sim * 0.1 +
                            expr_sim * 0.6) * 2.0
                else:
                    weight = (hic_weight * 0.3 +
                            compartment_sim * 0.1 +
                            tad_sim * 0.1 +
                            ins_sim * 0.1 +
                            expr_sim * 0.4) * cluster_sim
                
                G.add_edge(gene1, gene2, weight=weight)
                i, j = self.node_map[gene1], self.node_map[gene2]
                edge_index.extend([[i, j], [j, i]])
                edge_weights.extend([weight, weight])
          
            # Create Node2Vec embeddings with cluster-aware weights
            node2vec = Node2Vec(
                G,
                dimensions=self.embedding_dim,
                walk_length=10,
                num_walks=25,
                p=1.0,
                q=1.0,
                #p=1.739023,
                #q=1.6722,
                workers=1,
                seed=42
            )
            
            model = node2vec.fit(
                window=5,
                min_count=1,
                batch_words=4
            )

            # Create feature vectors
            features = []
            for gene in self.node_map.keys():
                # Get Node2Vec embedding
                node_embedding = torch.tensor(model.wv[gene], dtype=torch.float32)

                print(f"\n{gene} embedding analysis:")
                print(f"Original last dimension value: {node_embedding[-1]:.4f}")

                orig_mean = node_embedding.mean().item()
                orig_std = node_embedding.std().item()
                
                # Normalize embedding
                #node_embedding = (node_embedding - node_embedding.min()) / (node_embedding.max() - node_embedding.min() + 1e-8) #FIXME Normalization of embeddings not affect performance in a good way
                #print(f"Normalized last dimension value: {node_embedding[-1]:.4f}")
                print(f"Expression value to be inserted: {expression_values[gene]:.4f}")
                orig_last_dim = node_embedding[-1].item()
                
                # Last dimension is the normalized expression value for mRNA data because there are very low valued expressions
                if gene in ["AMACR", "ABCG2", "HPGDS"]: 
                    node_embedding[-1] = expression_values[gene] * 2.0 #scale negative correlated genes more higher value
                    print(f"Expression value of embedding from negative corel {gene}: {node_embedding[-1]}")
                else:
                    print(f"Node embeddings last dim before adding expression value: {node_embedding[-1]}")
                    node_embedding[-1] = expression_values[gene]
                    print(f"Node embeddings last dim after adding expression value: {node_embedding[-1]}")
                    print(f"Expression value for {gene}: {node_embedding[-1]}")
                
                print(f"Statistics before override:")
                print(f"  Mean: {orig_mean:.4f}")
                print(f"  Std: {orig_std:.4f}")
                print(f"  Last dim: {orig_last_dim:.4f}")
                print(f"Statistics after override:")
                print(f"  Mean: {node_embedding.mean().item():.4f}")
                print(f"  Std: {node_embedding.std().item():.4f}")
                print(f"  Last dim: {node_embedding[-1].item():.4f}")
                print('*************************************************************')
               
                features.append(node_embedding)
            
            temporal_features[t] = torch.stack(features)
            temporal_edge_indices[t] = torch.tensor(edge_index).t().contiguous()
            temporal_edge_attrs[t] = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)

            print(f"\nFeature Statistics for time {t}:")
            print(f"Expression range: [{min(expression_values.values()):.4f}, {max(expression_values.values()):.4f}]")
            print(f"Embedding range: [{temporal_features[t].min():.4f}, {temporal_features[t].max():.4f}]")
            
        return temporal_features, temporal_edge_indices, temporal_edge_attrs
    

    def create_temporal_node_features_several_graphs_created_mirna(self, debug_mode=True):

        temporal_features = {}
        temporal_edge_indices = {}
        temporal_edge_attrs = {}

      
        
        print("\nNormalizing expression values across all time points...")
        all_expressions = []
        log_expressions = []

        for t in self.time_points:
            for gene in self.node_map.keys():
                gene1_expr = self.df[self.df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
                gene2_expr = self.df[self.df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
                expr_value = gene1_expr[0] if len(gene1_expr) > 0 else \
                            (gene2_expr[0] if len(gene2_expr) > 0 else 0.0)
                log_expr = np.log1p(expr_value + 1e-7)
                
                all_expressions.append(expr_value)
                log_expressions.append(log_expr)
                
        global_min = min(all_expressions)
        global_max = max(all_expressions)
        print(f"Global expression range: [{global_min:.4f}, {global_max:.4f}]") #Global expression range: [1.0000, 19592.0000] 

        for t in self.time_points:
            print(f"\nProcessing time point {t}")
            #print(f"All time points: {self.time_points}")
            
            expression_values = {}
            for gene in self.node_map.keys():
                gene1_expr = self.df[self.df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
                gene2_expr = self.df[self.df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
                
                if np.isnan(gene1_expr).any():
                    print(f"NaN detected in Gene1 expression for {gene} at time {t}")
                if np.isnan(gene2_expr).any():
                    print(f"NaN detected in Gene2 expression for {gene} at time {t}")

                expr_value = gene1_expr[0] if len(gene1_expr) > 0 else \
                            (gene2_expr[0] if len(gene2_expr) > 0 else 0.0)

                expression_values[gene] = (expr_value - global_min) / (global_max - global_min + 1e-8)

            G = nx.Graph()
            G.add_nodes_from(self.node_map.keys())

            edge_index = []
            edge_weights = []
            for _, row in self.df.iterrows():
                gene1 = row['Gene1_clean']
                gene2 = row['Gene2_clean']
                
                expr_sim = 1 / (1 + abs(expression_values[gene1] - expression_values[gene2]))

                if np.isnan(expr_sim):
                    print(f"NaN detected in expression similarity between {gene1} and {gene2} at time {t}")
                
                hic_weight = row['HiC_Interaction'] 
                
                if pd.isna(row['HiC_Interaction']):
                    print(f"HiC weight is NaN")
            
                compartment_sim = 1 if row['Gene1_Compartment'] == row['Gene2_Compartment'] else 0 

                tad_dist = abs(row['Gene1_TAD_Boundary_Distance'] - row['Gene2_TAD_Boundary_Distance'])
                tad_sim = 1 / (1 + tad_dist)
                ins_sim = 1 / (1 + abs(row['Gene1_Insulation_Score'] - row['Gene2_Insulation_Score']))

                # Added for debugging of miRNA data
                if np.isnan(hic_weight):
                    print(f"NaN detected in HiC weight for {gene1}-{gene2}")
                if np.isnan(compartment_sim):
                    print(f"NaN detected in compartment similarity for {gene1}-{gene2}")
                if np.isnan(tad_sim):
                    print(f"NaN detected in TAD similarity for {gene1}-{gene2}")
                if np.isnan(ins_sim):
                    print(f"NaN detected in insulation similarity for {gene1}-{gene2}")

                weight = (hic_weight * 0.3 +
                        compartment_sim * 0.1 +
                        tad_sim * 0.1 +
                        ins_sim * 0.1 +
                        expr_sim * 0.4)
                
                G.add_edge(gene1, gene2, weight=weight)
                i, j = self.node_map[gene1], self.node_map[gene2]
                edge_index.extend([[i, j], [j, i]])
                edge_weights.extend([weight, weight])
          
            node2vec = Node2Vec(
                G,
                dimensions=self.embedding_dim,
                walk_length=10,
                num_walks=25,
                p=1.0,
                q=1.0,
                workers=1,
                seed=42
            )
            
            model = node2vec.fit(
                window=5,
                min_count=1,
                batch_words=4
            )

            features = []
            for gene in self.node_map.keys():
                node_embedding = torch.tensor(model.wv[gene], dtype=torch.float32)
                min_val = node_embedding.min()
                print(f"Node embedding original value: {node_embedding}")

                print(f"\n{gene} embedding analysis:")
                print(f"Original last dimension value: {node_embedding[-1]:.4f}")

                orig_mean = node_embedding.mean().item()
                orig_std = node_embedding.std().item()

                if min_val < 0:
                    node_embedding = node_embedding - min_val  
                    node_embedding = node_embedding / (node_embedding.max() + 1e-8)
                print(f"Node embedding value shifting through zero and normalized: {node_embedding}")
                
                # Normalize embedding
                #node_embedding = (node_embedding - node_embedding.min()) / (node_embedding.max() - node_embedding.min() + 1e-8) #FIXME Normalization of embeddings not affect performance in a good way
                #print(f"Normalized last dimension value: {node_embedding[-1]:.4f}")
                print(f"Expression value to be inserted: {expression_values[gene]:.4f}")
                orig_last_dim = node_embedding[-1].item()
    
                print(f"Node embeddings last dim before adding expression value: {node_embedding[-1]}")
                node_embedding[-1] = expression_values[gene]
                print(f"Node embeddings last dim after adding expression value: {node_embedding[-1]}")
                print(f"Expression value for {gene}: {node_embedding[-1]}")
                
                print(f"Statistics before override:")
                print(f"  Mean: {orig_mean:.4f}")
                print(f"  Std: {orig_std:.4f}")
                print(f"  Last dim: {orig_last_dim:.4f}")
                print(f"Statistics after override:")
                print(f"  Mean: {node_embedding.mean().item():.4f}")
                print(f"  Std: {node_embedding.std().item():.4f}")
                print(f"  Last dim: {node_embedding[-1].item():.4f}")
                print('*************************************************************')
               
                features.append(node_embedding)
            
            temporal_features[t] = torch.stack(features)
            temporal_edge_indices[t] = torch.tensor(edge_index).t().contiguous()
            temporal_edge_attrs[t] = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)

            print(f"\nFeature Statistics for time {t}:")
            print(f"Expression range: [{min(expression_values.values()):.4f}, {max(expression_values.values()):.4f}]")
            print(f"Embedding range: [{temporal_features[t].min():.4f}, {temporal_features[t].max():.4f}]")
            
        return temporal_features, temporal_edge_indices, temporal_edge_attrs
    
    def create_temporal_node_features_several_graphs_mirna_developed(self, debug_mode=True):
        temporal_features = {}
        temporal_edge_indices = {}
        temporal_edge_attrs = {}

    
        if debug_mode: print("\nCollecting expression values for transformation...")
        
        all_expressions = []
        expression_dict = {}  
        
        for t in self.time_points:
            expression_dict[t] = {}
            for gene in self.node_map.keys():
                gene1_expr = self.df[self.df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
                gene2_expr = self.df[self.df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
                expr_value = gene1_expr[0] if len(gene1_expr) > 0 else \
                            (gene2_expr[0] if len(gene2_expr) > 0 else 0.0)
                all_expressions.append(expr_value)
                expression_dict[t][gene] = expr_value

        transformer = QuantileTransformer(
            output_distribution='normal',
            n_quantiles=min(len(all_expressions), 1000),
            random_state=42
        )
        
        all_expressions = np.array(all_expressions).reshape(-1, 1)
        transformed_expressions = transformer.fit_transform(all_expressions)
   
        transformed_expressions = (transformed_expressions - transformed_expressions.min()) / \
                                (transformed_expressions.max() - transformed_expressions.min() + 1e-8) * 2 - 1
        
        if debug_mode:
            print(f"Original expression range: [{np.min(all_expressions):.4f}, {np.max(all_expressions):.4f}]")
            print(f"Transformed expression range: [{np.min(transformed_expressions):.4f}, {np.max(transformed_expressions):.4f}]")
            print(f"Standard deviation before: {np.std(all_expressions):.4f}")
            print(f"Standard deviation after: {np.std(transformed_expressions):.4f}")

        expr_idx = 0
        for t in self.time_points:
            if debug_mode: print(f"\nProcessing time point {t}...")

            expression_values = {}
            for gene in self.node_map.keys():
                expression_values[gene] = transformed_expressions[expr_idx][0]
                expr_idx += 1
            G = nx.Graph()
            G.add_nodes_from(self.node_map.keys())

            edge_index = []
            edge_weights = []

            current_expressions = np.array(list(expression_values.values()))
            high_threshold = np.percentile(current_expressions, 75)
            low_threshold = np.percentile(current_expressions, 25)
            
            for _, row in self.df.iterrows():
                gene1, gene2 = row['Gene1_clean'], row['Gene2_clean']
                if gene1 == gene2:
                    continue

                expr1 = expression_values[gene1]
                expr2 = expression_values[gene2]
                
                if expr1 > high_threshold and expr2 > high_threshold:
                    expr_sim = 0.9 * (1 / (1 + abs(expr1 - expr2)))
                elif expr1 < low_threshold and expr2 < low_threshold:
                    expr_sim = 0.3 * (1 / (1 + abs(expr1 - expr2)))
                else:
                    expr_sim = 0.6 * (1 / (1 + abs(expr1 - expr2)))

                hic_weight = row['HiC_Interaction']
                compartment_sim = 1 if row['Gene1_Compartment'] == row['Gene2_Compartment'] else 0 
                tad_sim = 1 / (1 + abs(row['Gene1_TAD_Boundary_Distance'] - row['Gene2_TAD_Boundary_Distance']))
                ins_sim = 1 / (1 + abs(row['Gene1_Insulation_Score'] - row['Gene2_Insulation_Score']))

                weight = (
                    hic_weight * 0.3 +
                    compartment_sim * 0.15 +
                    tad_sim * 0.15 +
                    ins_sim * 0.15 +
                    expr_sim * 0.25
                )

                if weight > 0.05:  # Lower threshold to keep more edges
                    G.add_edge(gene1, gene2, weight=weight)
                    i, j = self.node_map[gene1], self.node_map[gene2]
                    edge_index.extend([[i, j], [j, i]])
                    edge_weights.extend([weight, weight])

            node2vec = Node2Vec(
                G, 
                dimensions=self.embedding_dim, 
                walk_length=30,
                num_walks=20, 
                p=1.2,
                q=0.8,
                workers=1,
                seed=42
            )
            model = node2vec.fit(window=5, min_count=1, batch_words=4)

            features = []
            embedding_values = []
            for gene in self.node_map.keys():
                node_embedding = torch.tensor(model.wv[gene], dtype=torch.float32)
                embedding_values.append(node_embedding)
            embedding_values = torch.stack(embedding_values)

            embedding_values = (embedding_values - embedding_values.min()) / \
                            (embedding_values.max() - embedding_values.min() + 1e-8)

            for i, gene in enumerate(self.node_map.keys()):
                node_embedding = embedding_values[i]
                node_embedding[-1] = expression_values[gene]
                features.append(node_embedding)

            temporal_features[t] = torch.stack(features)
            temporal_edge_indices[t] = torch.tensor(edge_index).t().contiguous()
            temporal_edge_attrs[t] = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)

            if debug_mode:
                features_np = temporal_features[t].numpy()
                print(f"\nFeature Statistics for time {t}:")
                print(f"Feature matrix shape: {features_np.shape}")
                print(f"Feature min value: {features_np.min():.4f}")
                print(f"Feature max value: {features_np.max():.4f}")
                print(f"Feature std: {features_np.std():.4f}")

        return temporal_features, temporal_edge_indices, temporal_edge_attrs
    
    def create_temporal_node_features_several_graphs_created_clustering_closeness_betweenness(self):
        temporal_features = {}
        temporal_edge_indices = {}
        temporal_edge_attrs = {}

        clusters, _ = analyze_expression_levels_research(self)
        gene_clusters = {}
        for cluster_name, genes in clusters.items():
            for gene in genes:
                gene_clusters[gene] = cluster_name
        
        print("\nNormalizing expression values across all time points...")
        all_expressions = []
        for t in self.time_points:
            for gene in self.node_map.keys():
                gene1_expr = self.df[self.df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
                gene2_expr = self.df[self.df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
                expr_value = gene1_expr[0] if len(gene1_expr) > 0 else \
                            (gene2_expr[0] if len(gene2_expr) > 0 else 0.0)
                all_expressions.append(expr_value)
        
        global_min = min(all_expressions)
        global_max = max(all_expressions)
        print(f"Global expression range: [{global_min:.4f}, {global_max:.4f}]")
        
        for t in self.time_points:
            print(f"\nProcessing time point {t}")
            
            expression_values = {}
            for gene in self.node_map.keys():
                gene1_expr = self.df[self.df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
                gene2_expr = self.df[self.df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
                expr_value = gene1_expr[0] if len(gene1_expr) > 0 else \
                            (gene2_expr[0] if len(gene2_expr) > 0 else 0.0)
                expression_values[gene] = (expr_value - global_min) / (global_max - global_min)
            
            G = nx.Graph()
            G.add_nodes_from(self.node_map.keys())
        
            edge_index = []
            edge_weights = []

            for _, row in self.df.iterrows():
                gene1 = row['Gene1_clean']
                gene2 = row['Gene2_clean']
                
                hic_weight = row['HiC_Interaction']
                compartment_sim = 1 if row['Gene1_Compartment'] == row['Gene2_Compartment'] else 0

                initial_weight = hic_weight * 0.7 + compartment_sim * 0.3
                G.add_edge(gene1, gene2, weight=initial_weight)

            closeness = nx.closeness_centrality(G, distance='weight')
            max_close = max(closeness.values())

            if max_close == 0:
                print("Warning: All closeness centralities are zero, using raw values")

            norm_closeness = {k: v/max_close for k, v in closeness.items()}

            print(f"\nCloseness Statistics for time point {t}:")
            print(f"Closeness range: [{min(closeness.values()):.4f}, {max(closeness.values()):.4f}]")

            edge_index = []
            edge_weights = []
            for _, row in self.df.iterrows():
                gene1 = row['Gene1_clean']
                gene2 = row['Gene2_clean']
                
                expr_sim = 1 / (1 + abs(expression_values[gene1] - expression_values[gene2]))
                hic_weight = row['HiC_Interaction']
                compartment_sim = 1 if row['Gene1_Compartment'] == row['Gene2_Compartment'] else 0
                tad_dist = abs(row['Gene1_TAD_Boundary_Distance'] - row['Gene2_TAD_Boundary_Distance'])
                tad_sim = 1 / (1 + tad_dist)
                ins_sim = 1 / (1 + abs(row['Gene1_Insulation_Score'] - row['Gene2_Insulation_Score']))
                
                closeness_sim = (norm_closeness[gene1] + norm_closeness[gene2])/2
                
                cluster_sim = 1.2 if gene_clusters[gene1] == gene_clusters[gene2] else 1.0
                if gene1 in ["AMACR", "ABCG2", "HPGDS"] or gene2 in ["AMACR", "ABCG2", "HPGDS"]:
                    weight = (hic_weight * 0.15 +  
                            compartment_sim * 0.1 +
                            tad_sim * 0.1 +
                            ins_sim * 0.1 +
                            expr_sim * 0.4 +
                            closeness_sim * 0.15) * 2.0
                else:
                    weight = (hic_weight * 0.25 +
                            compartment_sim * 0.1 +
                            tad_sim * 0.1 +
                            ins_sim * 0.1 +
                            expr_sim * 0.3 +
                            closeness_sim * 0.15) * cluster_sim
                
                G.add_edge(gene1, gene2, weight=weight)
                i, j = self.node_map[gene1], self.node_map[gene2]
                edge_index.extend([[i, j], [j, i]])
                edge_weights.extend([weight, weight])
            
            node2vec = Node2Vec(
                G,
                dimensions=self.embedding_dim,
                walk_length=10,
                num_walks=25,
                p=1.0,
                q=1.0,
                workers=1,
                seed=42
            )
            
            model = node2vec.fit(
                window=5,
                min_count=1,
                batch_words=4
            )
            
            features = []
            for gene in self.node_map.keys():
                node_embedding = torch.tensor(model.wv[gene], dtype=torch.float32)
                
                node_embedding = (node_embedding - node_embedding.min()) / (node_embedding.max() - node_embedding.min() + 1e-8)
                
                if gene in ["AMACR", "ABCG2", "HPGDS"]: 
                    node_embedding[-1] = expression_values[gene] * 2.0
                else:
                    node_embedding[-1] = expression_values[gene]
                    
                features.append(node_embedding)
            
            temporal_features[t] = torch.stack(features)
            temporal_edge_indices[t] = torch.tensor(edge_index).t().contiguous()
            temporal_edge_attrs[t] = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)

            print(f"\nFeature Statistics for time {t}:")
            print(f"Expression range: [{min(expression_values.values()):.4f}, {max(expression_values.values()):.4f}]")
            print(f"Embedding range: [{temporal_features[t].min():.4f}, {temporal_features[t].max():.4f}]")
            
        return temporal_features, temporal_edge_indices, temporal_edge_attrs

    def get_edge_index_and_attr(self):
        """Convert base graph to PyG format"""
        edge_index = []
        edge_weights = []
        
        for u, v, d in self.base_graph.edges(data=True):
            i, j = self.node_map[u], self.node_map[v]
            edge_index.extend([[i, j], [j, i]])
            edge_weights.extend([d['weight'], d['weight']])
        
        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float32)
        edge_attr = edge_weights.unsqueeze(1)
        
        return edge_index, edge_attr
    
    def get_pyg_graph(self, time_point):
        """Create PyG graph for a specific time point"""
        return Data(
            x=self.node_features[time_point],
            edge_index=self.temporal_edge_indices[time_point],
            edge_attr=self.temporal_edge_attrs[time_point],
            num_nodes=self.num_nodes
        )
    
    def get_temporal_sequences(self):
        sequences = []
        labels = []
    
        #print("\nAvailable time points:", self.time_points)
        
        # First, create a clean dictionary of gene expressions across time
        gene_expressions = {}
        for t in self.time_points:
            # Convert time point to string for column access
            time_col = f'Gene1_Time_{t}'
            gene_expressions[t] = {}
            
            #print(f"\nProcessing time point {t}")  # Debug print
            
            for gene in self.node_map.keys():
                # Use the correct column name format
                gene1_expr = self.df[self.df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
                gene2_expr = self.df[self.df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
                
                # Take the first non-empty value
                if len(gene1_expr) > 0:
                    expr_value = gene1_expr[0]
                elif len(gene2_expr) > 0:
                    expr_value = gene2_expr[0]
                else:
                    print(f"Warning: No expression found for gene {gene} at time {t}")
                    expr_value = 0.0
                    
                gene_expressions[t][gene] = expr_value

        print("\nExpression value check:")
        for t in self.time_points[:5]:  # First 5 time points
            print(f"\nTime point {t}:")
            for gene in list(self.node_map.keys())[:5]:  # First 5 genes
                print(f"Gene {gene}: {gene_expressions[t][gene]}")
        
        # Create sequences
        for i in range(len(self.time_points) - self.seq_len - self.pred_len + 1):
            input_times = self.time_points[i:i+self.seq_len]
            target_times = self.time_points[i+self.seq_len:i+self.seq_len+self.pred_len]
            
            print(f"\nSequence {i}:")
            print(f"Input times: {input_times}")
            print(f"Target times: {target_times}")
            
            # Create sequence graphs
            seq_graphs = []
            for t in input_times:
                # The features should already be 32-dimensional from Node2Vec
                features = torch.tensor([gene_expressions[t][gene] for gene in self.node_map.keys()], 
                                    dtype=torch.float32)
                graph = self.get_pyg_graph(t)
                # Node2Vec features are already shape [num_nodes, 32]
                #print(f"Graph features shape: {graph.x.shape}")  # Should be [52, 32]
                seq_graphs.append(graph)
            
            # Create label graphs
            label_graphs = []
            for t in target_times:
                graph = self.get_pyg_graph(t)
                label_graphs.append(graph)
            
            sequences.append(seq_graphs)
            labels.append(label_graphs)
        
        return sequences, labels
    
    def split_sequences(self,sequences, labels):
        torch.manual_seed(42)
        
        n_samples = len(sequences)
        n_train = int(n_samples * (1 - 0.2))

        indices = torch.randperm(n_samples)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        train_sequences = [sequences[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_sequences = [sequences[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        
        print("\nData Split Statistics:")
        print(f"Total sequences: {n_samples}")
        print(f"Training sequences: {len(train_sequences)} ({len(train_sequences)/n_samples:.1%})")
        print(f"Validation sequences: {len(val_sequences)} ({len(val_sequences)/n_samples:.1%})")
        
        return train_sequences, train_labels, val_sequences, val_labels, train_idx, val_idx
        

 