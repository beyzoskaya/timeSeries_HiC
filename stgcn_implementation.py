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

import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

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

        print("Initial data shape:", self.df.shape)
        print("\nSample of time columns:", self.time_cols[:5])
        print("\nSample of unique genes:", list(self.node_map.keys())[:5])
        print("\nEdge index shape:", self.edge_index.shape)
        print("Edge weight shape:", self.edge_weight.shape)
        
        self.process_node_features()
    
    def process_node_features(self):

        print("\nProcessing node features...")
        print("Number of nodes:", len(self.node_map))

        self.node_features = {}

        for t in self.time_points:
            features = []
            for gene in self.node_map.keys():
                gene_data = self.df[(self.df['Gene1'] == gene) | (self.df['Gene2'] == gene)].iloc[0]
                print(f"\nFeatures for gene {gene}:")
                print(f"Expression: {gene_data[f'Gene1_Time_{t}']}")
                print(f"Compartment: {gene_data['Gene1_Compartment']}")
                print(f"TAD: {gene_data['Gene1_TAD_Boundary_Distance']}")
                print(f"Insulation: {gene_data['Gene1_Insulation_Score']}")

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

class STGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(STGCNLayer, self).__init__()
        self.temporal_conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.spatial_conv = GCNConv(out_channels, out_channels)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        
    def forward(self, x, edge_index, edge_weight):
        # Temporal convolution
        x = x.permute(0, 2, 1)  # [batch, features, time]
        x = self.temporal_conv(x)
        x = x.permute(0, 2, 1)  # [batch, time, features]
        
        # Spatial convolution for each time step
        batch_size, seq_len, num_features = x.size()
        x = x.reshape(-1, num_features)
        
        # Apply GCN
        x = self.spatial_conv(x, edge_index, edge_weight)
        x = self.batch_norm(x)
        x = F.relu(x)
        
        # Reshape back
        x = x.reshape(batch_size, seq_len, -1)
        return x

class STGCN(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels, num_layers=3):
        super(STGCN, self).__init__()
        self.num_layers = num_layers
        
        # Input layer
        self.input_layer = STGCNLayer(in_channels, hidden_channels)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            STGCNLayer(hidden_channels, hidden_channels)
            for _ in range(num_layers-2)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, edge_weight):
        # Input layer
        x = self.input_layer(x, edge_index, edge_weight)
        
        # Hidden layers
        for layer in self.hidden_layers:
            x = layer(x, edge_index, edge_weight)
        
        # Output layer
        x = self.output_layer(x)
        return x


def visualize_and_analyze_data(dataset):
    
    G = nx.Graph()
    edge_index = dataset.edge_index.numpy()
    edge_weight = dataset.edge_weight.numpy()
    
    for i in range(len(edge_index[0])):
        G.add_edge(edge_index[0][i], edge_index[1][i], 
                  weight=edge_weight[i])
    
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_size=50, with_labels=False, 
            edge_color='gray', alpha=0.5)
    plt.title(f"Graph Structure: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    plt.show()
    plt.savefig('plottings/graph_structure_stgcn.png')
    
    print("\nDataset Statistics:")
    print(f"Number of nodes: {dataset.num_nodes}")
    print(f"Number of edges: {len(dataset.edge_index[0])}")
    print(f"Number of time points: {len(dataset.time_points)}")
    print(f"Time points range: {min(dataset.time_points)} to {max(dataset.time_points)}")
    
    plt.figure(figsize=(15, 5))
    
    first_time = dataset.time_points[0]
    features = dataset.node_features[first_time].numpy()
    
    for i, name in enumerate(['Expression', 'Compartment', 'TAD_Distance', 'Insulation']):
        plt.subplot(1, 4, i+1)
        plt.hist(features[:, i], bins=30)
        plt.title(f'{name} Distribution')
    plt.tight_layout()
    plt.show()
    plt.savefig('plottings/feature_distribution.png')
    
    plt.figure(figsize=(12, 6))
    sample_nodes = np.random.choice(dataset.num_nodes, 5)  
    
    for node in sample_nodes:
        expressions = [dataset.node_features[t][node, 0].item() 
                      for t in dataset.time_points]
        plt.plot(dataset.time_points, expressions, label=f'Node {node}')
    
    plt.xlabel('Time')
    plt.ylabel('Expression')
    plt.title('Temporal Expression Patterns')
    plt.legend()
    plt.show()
    plt.savefig('plottings/temp_patterns.png')
    
    sequences, labels = dataset.get_temporal_sequences()
    print("\nSequence Information:")
    print(f"Number of sequences: {len(sequences)}")
    print(f"Sequence shape: {sequences[0].shape}")
    print(f"Label shape: {labels[0].shape}")
    
    plt.figure(figsize=(8, 5))
    plt.hist(dataset.edge_weight.numpy(), bins=50)
    plt.title('Edge Weight Distribution')
    plt.xlabel('Weight')
    plt.ylabel('Count')
    plt.show()


if __name__ == "__main__":
    dataset = TemporalGraphDataset('mapped/enhanced_interactions.csv', seq_len=10, pred_len=1)
    
    visualize_and_analyze_data(dataset)
    
    model = STGCN(
        num_nodes=dataset.num_nodes,
        in_channels=4,
        hidden_channels=64,
        out_channels=4,
        num_layers=3
    )
    print("\nModel Architecture:")
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal number of parameters: {total_params}")
