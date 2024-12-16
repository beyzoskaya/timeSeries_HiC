import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class TemporalGraphDataset:
    def __init__(self, csv_file, seq_len=10, pred_len=1):
        """
        seq_len --> number of time points to use for input
        pred_len --> number of time points to predict
        """
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.df = pd.read_csv(csv_file)
        self.process_data()
    
    def process_data(self):

        # need to get time columns
        self.time_cols = [col for col in self.df.columns if "Time_" in col]
        self.time_points = sorted(list(set([float(col.split('_')[-1]) for col in self.time_cols if 'Gene1' in col])))

        unique_genes = pd.concat([self.df['Gene1'], self.df['Gene2']]).unique()
        self.node_map = {gene: idx for idx, gene in enumerate(unique_genes)}
        self.num_nodes = len(self.node_map)

        # between two genes (Gene1 and Gene2)
        source_nodes = [self.node_map[gene] for gene in self.df['Gene1']]
        target_nodes = [self.node_map[gene] for gene in self.df['Gene2']] 

        self.edge_index = torch.tensor([source_nodes + target_nodes, 
                                      target_nodes + source_nodes], dtype=torch.long)
        
        self.edge_weight = torch.tensor(np.concatenate([self.df['HiC_Interaction'].values,
                                                      self.df['HiC_Interaction'].values]))
        self.edge_weight = (self.edge_weight - self.edge_weight.mean()) / self.edge_weight.std()
        
        self.process_node_features()
    
    def process_node_features(self):
        self.node_features = {}

        for t in self.time_points:
            features = []
            for gene in self.node_map.keys():
                gene_data = self.df[(self.df['Gene1'] == gene) | (self.df['Gene2'] == gene)].iloc[0]
                print(f"Gene data for {gene}: {gene_data}")

                expr = gene_data[f'Gene1_Time_{t}'] if gene == gene_data['Gene1'] else gene_data[f'Gene2_Time_{t}']

                # get compartments as one-hot encoded version
                comp = 1 if (gene == gene_data['Gene1'] and gene_data['Gene1_Compartment'] == 'A') or \
                          (gene == gene_data['Gene2'] and gene_data['Gene2_Compartment'] == 'A') else 0
                
                # get TAD boundries and insulation features
                tad = gene_data['Gene1_TAD_Boundary_Distance'] if gene== gene_data['Gene1'] else gene_data['Gene2_TAD_Boundary_Distance']
                ins = gene_data['Gene1_Insulation_Score'] if gene == gene_data['Gene1'] else gene_data['Gene2_Insulation_Score']
                
                features.append([expr, comp, tad, ins])
            
            features = torch.tensor(features, dtype=torch.float)
            self.node_features[t] = features
        
    def get_temporal_sequences(self):
        sequences = []
        labels = []

        for i in range(len(self.time_points) - self.seq_len - self.pred_len + 1):

            seq_times = self.time_points[i:i+self.seq_len]
            seq_data = [self.node_features[t] for t in seq_times]

            target_times = self.time_points[i+self.seq_len:i+self.seq_len+self.pred_len]
            target_data = [self.node_features[t] for t in target_times]

            sequences.append(torch.stack(seq_data))
            labels.append(torch.stack(target_data))
        return sequences, labels