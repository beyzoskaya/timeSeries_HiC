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
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from torch.utils.data import TensorDataset, DataLoader


class TemporalGraphDatasetSlidingWindow:
    def __init__(self, csv_file, embedding_dim=32, seq_len=3, pred_len=1, graph_params=None, node2vec_params=None):
    
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.embedding_dim = embedding_dim
        
        self.df = pd.read_csv(csv_file)
        print(f"Before cleaning the gene names Gene1: {self.df['Gene1']}")
        print(f"Before cleaning the gene names Gene2: {self.df['Gene2']}")
        
        self.df['Gene1_clean'] = self.df['Gene1']
        self.df['Gene2_clean'] = self.df['Gene2']
        self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed', case=False)]

        unique_genes = pd.concat([self.df['Gene1_clean'], self.df['Gene2_clean']]).unique()
        self.node_map = {gene: idx for idx, gene in enumerate(unique_genes)}
        print(f"Unique genes: {self.node_map}")
        self.num_nodes = len(self.node_map)
        print(f"Number of nodes: {self.num_nodes}")
      
        self.time_cols = [col for col in self.df.columns if 'Time_' in col]
        self.time_points = sorted(list(set([float(col.split('_')[-1])  
                                          for col in self.time_cols if 'Gene1' in col])))
        self.time_points = [float(tp) for tp in self.time_points]
      
        print(f"Found {len(self.time_points)} time points")
        print("Extracted time points:", self.time_points)
        
        self.base_graph = self.create_base_graph()
        print("Base graph created")
        self.node_features, self.temporal_edge_indices, self.temporal_edge_attrs = \
        self.create_temporal_node_features_several_graphs_created_mirna(debug_mode=True)
        print("Temporal node features created")
        
        self.edge_index, self.edge_attr = self.get_edge_index_and_attr()
        print(f"Graph structure created with {len(self.edge_attr)} edges")
    
    def create_base_graph(self):
        G = nx.Graph()
        G.add_nodes_from(self.node_map.keys())
        
        for _, row in self.df.iterrows():
            hic_weight = row['HiC_Interaction']
            compartment_sim = 1 if row['Gene1_Compartment'] == row['Gene2_Compartment'] else 0
            tad_dist = abs(row['Gene1_TAD_Boundary_Distance'] - row['Gene2_TAD_Boundary_Distance'])
            tad_sim = 1 / (1 + tad_dist)
            ins_sim = 1 / (1 + abs(row['Gene1_Insulation_Score'] - row['Gene2_Insulation_Score']))
            
            weight = (hic_weight * 0.4 + 
                     compartment_sim * 0.2 + 
                     tad_sim * 0.2 + 
                     ins_sim * 0.2)
            
            G.add_edge(row['Gene1_clean'], row['Gene2_clean'], weight=weight)
        
        return G
        
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
                #softplus for node embedding's negative values
                #node_embedding = torch.log(1 + torch.exp(node_embedding))
                min_val = node_embedding.min()
                print(f"Node embedding original value: {node_embedding}")

                print(f"\n{gene} embedding analysis:")
                print(f"Original last dimension value: {node_embedding[-1]:.4f}")

                if min_val < 0:
                    node_embedding = node_embedding - min_val  
                    node_embedding = node_embedding / (node_embedding.max() + 1e-8)
                print(f"Node embedding value shifting through zero and normalized: {node_embedding}")

                orig_mean = node_embedding.mean().item()
                orig_std = node_embedding.std().item()
                
                # Normalize embedding
                #node_embedding = (node_embedding - node_embedding.min()) / (node_embedding.max() - node_embedding.min() + 1e-8) #FIXME Normalization of embeddings not affect performance in a good way
                #print(f"Normalized last dimension value: {node_embedding[-1]:.4f}")

                #print(f"Expression value to be inserted: {expression_values[gene]:.4f}")

                orig_last_dim = node_embedding[-1].item()
    
                #print(f"Node embeddings last dim before adding expression value: {node_embedding[-1]}")
                #node_embedding[-1] = expression_values[gene]
                #print(f"Node embeddings last dim after adding expression value: {node_embedding[-1]}")
                #print(f"Expression value for {gene}: {node_embedding[-1]}")
                
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

    def get_edge_index_and_attr(self):
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

        return Data(
            x=self.node_features[time_point],
            edge_index=self.temporal_edge_indices[time_point],
            edge_attr=self.temporal_edge_attrs[time_point],
            num_nodes=self.num_nodes
        )
    
    def get_expression_matrix(self):

        n_nodes = self.num_nodes
        n_time_points = len(self.time_points)
        

        expression_matrix = torch.zeros((n_nodes, n_time_points), dtype=torch.float32)
        
        for t_idx, t in enumerate(self.time_points):
            for gene, node_idx in self.node_map.items():

                gene1_expr = self.df[self.df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
                gene2_expr = self.df[self.df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
                
                if len(gene1_expr) > 0:
                    expr_value = gene1_expr[0]
                elif len(gene2_expr) > 0:
                    expr_value = gene2_expr[0]
                else:
                    expr_value = 0.0
                
                expression_matrix[node_idx, t_idx] = expr_value
        
        return expression_matrix
    
    def get_temporal_sequences(self):

        expression_matrix = self.get_expression_matrix()  # [nodes, time_points]
     
        sequences = []  # Will contain input graph sequences
        labels = []     # Will contain target graph sequences
        time_info = []  # Will store time information for each sequence
        
        n_sequences = len(self.time_points) - self.seq_len - self.pred_len + 1
        
        for i in range(n_sequences):
            # Input window: [t, t+1, t+2]
            input_times = self.time_points[i:i+self.seq_len]
            # Target window: [t+3]
            target_times = self.time_points[i+self.seq_len:i+self.seq_len+self.pred_len]
            
            print(f"\nSequence {i}:")
            print(f"Input times: {input_times}")
            print(f"Target times: {target_times}")
            
            seq_graphs = []
            for t in input_times:
                t_idx = self.time_points.index(t)
                graph = self.get_pyg_graph(t)
                seq_graphs.append(graph)
            
            label_graphs = []
            for t in target_times:
                t_idx = self.time_points.index(t)
                graph = self.get_pyg_graph(t)
                label_graphs.append(graph)
            
            sequences.append(seq_graphs)
            labels.append(label_graphs)
            time_info.append({
                'input_times': input_times,
                'target_times': target_times
            })
        
        return sequences, labels, time_info
    
    def prepare_stgcn_data(self):

        expression_matrix = self.get_expression_matrix()  # [nodes, time_points]
        n_nodes = self.num_nodes
        n_time_points = len(self.time_points)
        
        X = []  # Input sequences
        y = []  # Target sequences
        sequence_info = []  # Time information
    
        n_sequences = len(self.time_points) - self.seq_len - self.pred_len + 1
        
        for i in range(n_sequences):
            # Input time indices: [t, t+1, t+2]
            input_indices = list(range(i, i + self.seq_len))
            # Target time indices: [t+3]
            target_indices = list(range(i + self.seq_len, i + self.seq_len + self.pred_len))

            input_times = [self.time_points[idx] for idx in input_indices]
            target_times = [self.time_points[idx] for idx in target_indices]
            
            print(f"\nSequence {i}:")
            print(f"Input times: {input_times}")
            print(f"Target times: {target_times}")
            

            input_tensor = expression_matrix[:, input_indices].t().unsqueeze(0)  # [1, seq_len, nodes]
            
            # [1, pred_len, nodes]
            target_tensor = expression_matrix[:, target_indices].t().unsqueeze(0)  # [1, pred_len, nodes]
            
            # [batch=1, features=1, time_steps, nodes]
            input_stgcn = input_tensor.unsqueeze(1)  # [1, 1, seq_len, nodes]
            target_stgcn = target_tensor.unsqueeze(1)  # [1, 1, pred_len, nodes]
            
            X.append(input_stgcn)
            y.append(target_stgcn)
            sequence_info.append({
                'input_times': input_times,
                'target_times': target_times
            })
        
        if X:
            X = torch.cat(X, dim=0)  # [n_sequences, 1, seq_len, nodes]
            y = torch.cat(y, dim=0)  # [n_sequences, 1, pred_len, nodes]
        else:
            print("Warning: No sequences were created!")
            X = torch.tensor([])
            y = torch.tensor([])
        
        return X, y, sequence_info
    
    def prepare_attention_model_data(self):

        expression_matrix = self.get_expression_matrix()  # [nodes, time_points]
        n_nodes = self.num_nodes
        n_time_points = len(self.time_points)
    
        X = []  # Input sequences
        y = []  # Target sequences
        sequence_info = []  # Time information
        
        n_sequences = len(self.time_points) - self.seq_len - self.pred_len + 1
        
        for i in range(n_sequences):
            # Input time indices: [t, t+1, t+2]
            input_indices = list(range(i, i + self.seq_len))
            # Target time indices: [t+3]
            target_indices = list(range(i + self.seq_len, i + self.seq_len + self.pred_len))
            
            # Get input and target times
            input_times = [self.time_points[idx] for idx in input_indices]
            target_times = [self.time_points[idx] for idx in target_indices]

            print(f"\nSequence {i}:")
            print(f"Input times: {input_times}")
            print(f"Target times: {target_times}")

            input_features = []
            for t in input_times:
                node_embeds = self.node_features[t]  # [nodes, embedding_dim]
                input_features.append(node_embeds)
            
            # [seq_len, nodes, embedding_dim]
            input_tensor = torch.stack(input_features, dim=0)
            
            #  [nodes, seq_len, embedding_dim]
            input_tensor = input_tensor.permute(1, 0, 2)
           
            target_expr = expression_matrix[:, target_indices].t()
     
            input_model = input_tensor.unsqueeze(0)  # [1, nodes, seq_len, embedding_dim]
            target_model = target_expr.unsqueeze(0).unsqueeze(1)  # [1, 1, pred_len, nodes]
            
            X.append(input_model)
            y.append(target_model)
            sequence_info.append({
                'input_times': input_times,
                'target_times': target_times
            })
        
        # Stack all sequences into a single batch
        if X:
           
            X = torch.cat(X, dim=0)  # [n_sequences, nodes, seq_len, embedding_dim]
            y = torch.cat(y, dim=0)  # [n_sequences, 1, pred_len, nodes]
        else:
            print("Warning: No sequences were created!")
            X = torch.tensor([])
            y = torch.tensor([])
        
        return X, y, sequence_info
    
    def split_sequences(self, X, y, train_ratio=0.8):

        torch.manual_seed(42)
        
        n_samples = X.shape[0]
        n_train = int(n_samples * train_ratio)

        indices = torch.randperm(n_samples)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        train_X = X[train_idx]
        train_y = y[train_idx]
        val_X = X[val_idx]
        val_y = y[val_idx]
        
        print("\nData Split Statistics:")
        print(f"Total sequences: {n_samples}")
        print(f"Training sequences: {len(train_X)} ({len(train_X)/n_samples:.1%})")
        print(f"Validation sequences: {len(val_X)} ({len(val_X)/n_samples:.1%})")
        
        return train_X, train_y, val_X, val_y, train_idx, val_idx
    
    def create_data_loaders(self, X, y, batch_size=32):

        train_X, train_y, val_X, val_y, train_idx, val_idx = self.split_sequences(X, y)
        
        train_dataset = TensorDataset(train_X, train_y)
        val_dataset = TensorDataset(val_X, val_y)
    
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        return train_loader, val_loader, train_idx, val_idx