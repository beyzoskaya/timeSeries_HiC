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
import sys
import argparse
from scipy.spatial.distance import cdist
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from networkx.algorithms.components import is_connected
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from torch.utils.data import DataLoader, TensorDataset
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class TemporalGraphDataset:
    def __init__(self, csv_file, embedding_dim=32, seq_len=4, pred_len=1):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.embedding_dim = embedding_dim
        
        self.df = pd.read_csv(csv_file)
        self.df['Gene1_clean'] = self.df['Gene1']
        self.df['Gene2_clean'] = self.df['Gene2']
        self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed', case=False)]

        unique_genes = pd.concat([self.df['Gene1_clean'], self.df['Gene2_clean']]).unique()
        self.node_map = {gene: idx for idx, gene in enumerate(unique_genes)}
        self.num_nodes = len(self.node_map)
        
        self.time_cols = [col for col in self.df.columns if 'Time_' in col]
        self.time_points = sorted(list(set([float(col.split('_')[-1])  
                                          for col in self.time_cols if 'Gene1' in col])))
        print(f"Found time points: {self.time_points}")
        print(f"Len of time points: {len(self.time_points)}")
        self.time_points = [tp for tp in self.time_points if tp != 154.0]
        self.df = self.df.loc[:, ~self.df.columns.str.contains('Time_154.0', case=False)]
        
        self.base_graph = self.create_base_graph()
        
        self.node_features, self.temporal_edge_indices, self.temporal_edge_attrs = \
            self.create_temporal_node_features_with_node2vec()
        
        self.edge_index, self.edge_attr = self.get_edge_index_and_attr()
    
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

    def create_temporal_node_features_with_node2vec(self):
        temporal_node_features = {}
        temporal_edge_indices = {}
        temporal_edge_attrs = {}

        for t in self.time_points:
            G = nx.Graph()
            G.add_nodes_from(self.node_map.keys())
            
            edge_index = []
            edge_weights = []
            
            for _, row in self.df.iterrows():
                gene1 = row['Gene1_clean']
                gene2 = row['Gene2_clean']
                
                # Use expression difference as similarity
                gene1_expr = row.get(f'Gene1_Time_{t}', 0.0)
                gene2_expr = row.get(f'Gene2_Time_{t}', 0.0)
                expr_sim = 1 / (1 + abs(gene1_expr - gene2_expr))
                print(f"Processing genes: {gene1}, {gene2} at time {t} with expr_sim: {expr_sim}")
                
                hic_weight = row['HiC_Interaction'] if not pd.isna(row['HiC_Interaction']) else 0
                compartment_sim = 1 if row['Gene1_Compartment'] == row['Gene2_Compartment'] else 0
                tad_dist = abs(row['Gene1_TAD_Boundary_Distance'] - row['Gene2_TAD_Boundary_Distance'])
                tad_sim = 1 / (1 + tad_dist)
                ins_sim = 1 / (1 + abs(row['Gene1_Insulation_Score'] - row['Gene2_Insulation_Score']))
                
                weight = (hic_weight * 0.25 +
                        compartment_sim * 0.1 +
                        tad_sim * 0.1 +
                        ins_sim * 0.1 +
                        expr_sim * 0.45)
            
                
                G.add_edge(gene1, gene2, weight=weight)
                
                i, j = self.node_map[gene1], self.node_map[gene2]
                edge_index.extend([[i, j], [j, i]])
                edge_weights.extend([weight, weight])
            
            # Node2Vec on this temporal graph
            node2vec = Node2Vec(
                G,
                dimensions=self.embedding_dim,
                walk_length=25,
                num_walks=75,
                p=1.0,
                q=1.0,
                workers=1,
                seed=42
            )
            model = node2vec.fit(window=5, min_count=1, batch_words=4)
            
            # Extract embeddings and convert to tensor
            embeddings = []
            for gene in self.node_map.keys():
                embeddings.append(model.wv[str(gene)] if str(gene) in model.wv else np.zeros(self.embedding_dim))
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
            
            temporal_node_features[t] = embeddings
            temporal_edge_indices[t] = torch.tensor(edge_index).t().contiguous()
            #temporal_edge_attrs[t] = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)
            temporal_edge_attrs[t] = torch.tensor(edge_weights, dtype=torch.float32)
        
        return temporal_node_features, temporal_edge_indices, temporal_edge_attrs

    def get_edge_index_and_attr(self):
        edge_index = []
        edge_weights = []
        for u, v, d in self.base_graph.edges(data=True):
            i, j = self.node_map[u], self.node_map[v]
            edge_index.extend([[i, j], [j, i]])
            edge_weights.extend([d['weight'], d['weight']])
        return (
            torch.tensor(edge_index).t().contiguous(),
            torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)
        )

    def get_pyg_graph(self, time_point):
        return Data(
            x=self.node_features[time_point],
            edge_index=self.temporal_edge_indices[time_point],
            edge_attr=self.temporal_edge_attrs[time_point],
            num_nodes=self.num_nodes
        )

    def get_temporal_sequences(self):
        sequences, labels = [], []
        for i in range(len(self.time_points) - self.seq_len - self.pred_len + 1):
            input_times = self.time_points[i:i+self.seq_len]
            target_times = self.time_points[i+self.seq_len:i+self.seq_len+self.pred_len]
            seq_graphs = [self.get_pyg_graph(t) for t in input_times]
            label_graphs = [self.get_pyg_graph(t) for t in target_times]
            sequences.append(seq_graphs)
            labels.append(label_graphs)
        return sequences, labels

    def split_sequences(self, sequences, labels):
        torch.manual_seed(42)
        n_samples = len(sequences)
        n_train = int(n_samples * 0.8)
        indices = torch.randperm(n_samples)
        train_idx, val_idx = indices[:n_train], indices[n_train:]
        train_sequences = [sequences[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_sequences = [sequences[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        return train_sequences, train_labels, val_sequences, val_labels, train_idx, val_idx
