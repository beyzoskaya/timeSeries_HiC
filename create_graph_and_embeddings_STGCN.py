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

def clean_gene_name(gene_name):
    """Clean gene name by removing descriptions and extra information"""
    if pd.isna(gene_name):
        return gene_name
    return gene_name.split('(')[0].strip()

class TemporalGraphDataset:
    def __init__(self, csv_file, embedding_dim=32, seq_len=5, pred_len=1): # I change the seq_len to more lower value
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.embedding_dim = embedding_dim
        
        self.df = pd.read_csv(csv_file)
        self.df['Gene1_clean'] = self.df['Gene1'].apply(clean_gene_name)
        self.df['Gene2_clean'] = self.df['Gene2'].apply(clean_gene_name)
        
        # Create static node mapping
        unique_genes = pd.concat([self.df['Gene1_clean'], self.df['Gene2_clean']]).unique()
        self.node_map = {gene: idx for idx, gene in enumerate(unique_genes)}
        self.num_nodes = len(self.node_map)
        print(f"Number of nodes: {self.num_nodes}")
        
        # Get time points
        self.time_cols = [col for col in self.df.columns if 'Time_' in col]
        self.time_points = sorted(list(set([float(col.split('_')[-1]) 
                                          for col in self.time_cols if 'Gene1' in col])))
        print(f"Found {len(self.time_points)} time points")
        
        # Create base graph and features
        self.base_graph = self.create_base_graph()
        print("Base graph created")
        self.node_features = self.create_temporal_node_features_several_graphs_created() # try with several graphs for time series consistency
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
        """Create temporal node features with correct dimensionality"""
        temporal_features = {}
        nodes = list(self.node_map.keys())

        print("\n=== Initial Data Statistics ===")
        for t in self.time_points:
            print(f"\nProcessing time point {t}")

            G = nx.Graph()
            G.add_nodes_from(self.node_map.keys())

            gene1_expr = self.df[f'Gene1_Time_{t}'].values
            gene2_expr = self.df[f'Gene2_Time_{t}'].values
            all_expr = np.concatenate([gene1_expr, gene2_expr])
            print(f"\nTime point {t}:")
            print(f"Expression range: [{np.min(all_expr):.4f}, {np.max(all_expr):.4f}]")
            print(f"Mean: {np.mean(all_expr):.4f}")
            print(f"Std: {np.std(all_expr):.4f}")
            
            # Create graph for this time point
            G = nx.Graph()
            G.add_nodes_from(self.node_map.keys())
            
            # Add edges with weights
            edge_weights = []
            for _, row in self.df.iterrows():
                gene1_expr = row[f'Gene1_Time_{t}']
                gene2_expr = row[f'Gene2_Time_{t}']
                
                # Calculate edge weights incorporating temporal information
                hic_weight = row['HiC_Interaction']
                compartment_sim = 1 if row['Gene1_Compartment'] == row['Gene2_Compartment'] else 0
                tad_dist = abs(row['Gene1_TAD_Boundary_Distance'] - row['Gene2_TAD_Boundary_Distance'])
                tad_sim = 1 / (1 + tad_dist)
                ins_sim = 1 / (1 + abs(row['Gene1_Insulation_Score'] - row['Gene2_Insulation_Score']))
                expr_sim = 1 / (1 + abs(gene1_expr - gene2_expr))
                
                # more temporal focus weights for new model
                weight = (hic_weight * 0.25 +  # 25% HiC
                compartment_sim * 0.1 +        # 10% Compartment
                tad_sim * 0.1 +                # 10% TAD
                ins_sim * 0.05 +               # 5% Insulation
                expr_sim * 0.5)
                edge_weights.append((row['Gene1_clean'], row['Gene2_clean'], weight))
     
            #G.add_edge(row['Gene1_clean'], row['Gene2_clean'], weight=weight)
                
            # Normalize edge weights to [0,1]
            weights = np.array([w for _, _, w in edge_weights])
            min_weight = np.min(weights)
            max_weight = np.max(weights)
            normalized_weights = (weights - min_weight) / (max_weight - min_weight)
            
            # Add normalized edges to graph
            for (gene1, gene2, _), norm_weight in zip(edge_weights, normalized_weights):
                G.add_edge(gene1, gene2, weight=norm_weight)
            
            edge_weights_dict = nx.get_edge_attributes(G, 'weight')
            min_weight = min(edge_weights_dict.values())
            max_weight = max(edge_weights_dict.values())

            print("\nGraph Statistics:")
            print(f"Number of edges: {G.number_of_edges()}")
            print(f"Edge weight range: [{min_weight:.4f}, {max_weight:.4f}]")
            # Create Node2Vec embeddings with exact embedding_dim dimensions
            node2vec = Node2Vec(
                G,
                dimensions=self.embedding_dim,  # Use full embedding dimension
                walk_length=20,
                num_walks=150,
                workers=1,
                p=1.0,
                q=1.0,
                weight_key='weight'
            )
            
            model = node2vec.fit(
            window=15,      # Increased from 10
            min_count=1,
            batch_words=4   # Added batch processing
        )
            expression_values = {}
            for gene in self.node_map.keys():
                gene1_expr = self.df[self.df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
                gene2_expr = self.df[self.df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
                expr_value = gene1_expr[0] if len(gene1_expr) > 0 else \
                            (gene2_expr[0] if len(gene2_expr) > 0 else 0.0)
                expression_values[gene] = expr_value
            
            # Normalize expression values
            expr_min = min(expression_values.values())
            expr_max = max(expression_values.values())
            normalized_expressions = {
                gene: (val - expr_min) / (expr_max - expr_min)
                for gene, val in expression_values.items()
            }
            
            # Create features for all nodes
            features = []
            for gene in self.node_map.keys():
                # Get Node2Vec embedding
                node_embedding = torch.tensor(model.wv[gene], dtype=torch.float32)
                # still I am seeing negative values for node embeddings
                node_embedding = (node_embedding - node_embedding.min()) / (node_embedding.max() - node_embedding.min())
                
                # Get expression value for this time point
                gene1_expr = self.df[self.df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
                gene2_expr = self.df[self.df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
                expr_value = gene1_expr[0] if len(gene1_expr) > 0 else \
                            (gene2_expr[0] if len(gene2_expr) > 0 else 0.0)
                
                # Modify last dimension of embedding to incorporate expression
                node_embedding[-1] = normalized_expressions[gene]
                
                features.append(node_embedding)
            
            # Stack features
            temporal_features[t] = torch.stack(features)

            #for t, features in temporal_features.items():
            #    print(f"Time {t}: Feature shape {features.shape}")

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

            #print(f"Sample Node2Vec embedding for a node: {temporal_features[self.time_points[0]][0, :5]}")

            # Verify dimensions for first time point
            if t == self.time_points[0]:
                print(f"\nFeature verification for time {t}:")
                print(f"Feature shape: {temporal_features[t].shape}")
                print(f"Expected shape: ({len(self.node_map)}, {self.embedding_dim})")
                
                # Verify content
                first_gene = list(self.node_map.keys())[0]
                print(f"\nExample features for {first_gene}:")
                print(f"Shape: {temporal_features[t][0].shape}")
                print(f"First 5 values: {temporal_features[t][0, :5]}")
                print(f"Expression value (last dim): {temporal_features[t][0, -1]}")
        
        return temporal_features

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
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            num_nodes=self.num_nodes
        )
    
    def get_temporal_sequences(self):
        """Create temporal sequences for training"""
        sequences = []
        labels = []
        print("\n=== Sequence Creation Statistics ===")
        print(f"Time points: {self.time_points}")
        print(f"Sequence length: {self.seq_len}")
        print(f"Prediction length: {self.pred_len}")

        #print("\nTime points being used:")
        #print(f"All time points: {self.time_points}")
        
        for i in range(len(self.time_points) - self.seq_len - self.pred_len + 1):
            seq_graphs = [self.get_pyg_graph(t) for t in self.time_points[i:i+self.seq_len]]
            label_graphs = [self.get_pyg_graph(t) for t in 
                          self.time_points[i+self.seq_len:i+self.seq_len+self.pred_len]]
            
            #print(f"Sequence graphs: {[g.x.shape for g in seq_graphs]}")
            #print(f"Label graphs: {[g.x.shape for g in label_graphs]}")

            #for graph in seq_graphs[:1]:  # Check only the first sequence graph
            #    print(graph.edge_index)
            
            #print(f"Feature mean: {torch.cat([g.x for g in seq_graphs]).mean(dim=0)}")
            #print(f"Feature std: {torch.cat([g.x for g in seq_graphs]).std(dim=0)}")
            
            #print(f"Label graphs (sample): {[g.x.shape for g in label_graphs[:3]]}")

            #input_times = self.time_points[i:i+self.seq_len]
            target_times = self.time_points[i+self.seq_len:i+self.seq_len+self.pred_len]
            
            if i == 0:
                print("\nSequence information:")
                print(f"Input time points: {self.time_points[i:i+self.seq_len]}")
                print(f"Target time points: {self.time_points[i+self.seq_len:i+self.seq_len+self.pred_len]}")
                print(f"Feature dimension: {seq_graphs[0].x.shape[1]}")

                # debugging for gene values
                #label_tensor = torch.stack([g.x for g in label_graphs]).mean(dim=0)
                label_tensor = torch.stack([g.x for g in label_graphs]).squeeze(dim=0) # Instead of mean, I directly squeeze the dim
                #print(f" Label tensor: {label_tensor}")
                #print(f" Label tensor shape: {label_tensor.shape}") # [1, 52, 32] without mean(dim=0)--> with dim=0 [52, 32] 
                genes = list(self.node_map.keys())
                print("\nSample label values for first 5 genes:")
                for idx in range(min(5, len(genes))):
                    gene = genes[idx]
                    value = label_tensor[idx]
                    print(f"{gene}: {value.mean().item():.4f}")
                
                # Check raw expression values
                #print("\nRaw expression values for first gene:")
                #first_gene = genes[0]
                #for t in target_times:
                #    expr = self.df[self.df['Gene1_clean'] == first_gene][f'Gene1_Time_{t}'].values
                #   if len(expr) == 0:
                #        expr = self.df[self.df['Gene2_clean'] == first_gene][f'Gene2_Time_{t}'].values
                #    print(f"Time {t}: {expr[0] if len(expr) > 0 else 'Not found'}")
            
            sequences.append(seq_graphs)
            labels.append(label_graphs)
            #print(f"Labels: {labels}")
        
        print(f"\nCreated {len(sequences)} sequences")

        return sequences, labels
    