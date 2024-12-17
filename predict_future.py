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
import os
import seaborn as sns

class TemporalGraphDataset:
    def __init__(self, csv_file, seq_len=10, pred_len=1):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.df = pd.read_csv(csv_file)
        self.process_data()
    
    def process_data(self):
        # Get time columns
        self.time_cols = [col for col in self.df.columns if 'Time_' in col]
        self.time_points = sorted(list(set([float(col.split('_')[-1]) 
                                          for col in self.time_cols if 'Gene1' in col])))
        
        # Create node mapping
        unique_genes = pd.concat([self.df['Gene1'], self.df['Gene2']]).unique()
        self.node_map = {gene: idx for idx, gene in enumerate(unique_genes)}
        self.num_nodes = len(self.node_map)
        print(f"Number of nodes: {self.num_nodes}")
        
        # Create graph structure from adjacency matrix
        self.G = self.create_graph_structure()
        
        # Process node features
        self.process_node_features()
    
    def create_adjacency_matrix(self):
        adj_matrix = np.zeros((self.num_nodes, self.num_nodes))
        for _, row in self.df.iterrows():
            i = self.node_map[row['Gene1']]
            j = self.node_map[row['Gene2']]
            adj_matrix[i, j] = row['HiC_Interaction']
            adj_matrix[j, i] = row['HiC_Interaction']
        return adj_matrix
    
    def create_graph_structure(self):
        adj_matrix = self.create_adjacency_matrix()
        G = nx.from_numpy_array(adj_matrix)
        
        edge_index = []
        edge_weights = []
        
        for u, v, d in G.edges(data=True):
            edge_index.append([u, v])
            edge_weights.append(d['weight'])
        
        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        
        # Normalize edge weights
        edge_weights = (edge_weights - edge_weights.mean()) / (edge_weights.std() + 1e-6)
        self.edge_index = edge_index
        self.edge_attr = edge_weights.unsqueeze(1)
        
        return G
    
    def process_node_features(self):
        """Process node features with normalization"""
        self.node_features = {}
        self.scalers = [StandardScaler() for _ in range(4)]
        
        # Collect all features for fitting scalers
        all_features = {i: [] for i in range(4)}
        for t in self.time_points:
            for gene in self.node_map.keys():
                gene_data = self.df[(self.df['Gene1'] == gene) | (self.df['Gene2'] == gene)].iloc[0]
                features = [
                    gene_data[f'Gene1_Time_{t}'] if gene == gene_data['Gene1'] else gene_data[f'Gene2_Time_{t}'],
                    1 if ((gene == gene_data['Gene1'] and gene_data['Gene1_Compartment'] == 'A') or
                         (gene == gene_data['Gene2'] and gene_data['Gene2_Compartment'] == 'A')) else 0,
                    gene_data['Gene1_TAD_Boundary_Distance'] if gene == gene_data['Gene1'] else gene_data['Gene2_TAD_Boundary_Distance'],
                    gene_data['Gene1_Insulation_Score'] if gene == gene_data['Gene1'] else gene_data['Gene2_Insulation_Score']
                ]
                for i, feat in enumerate(features):
                    all_features[i].append(feat)

        for i in range(4):
            if i != 1:  
                self.scalers[i].fit(np.array(all_features[i]).reshape(-1, 1))
        
        for t in self.time_points:
            features = []
            for gene in self.node_map.keys():
                gene_data = self.df[(self.df['Gene1'] == gene) | (self.df['Gene2'] == gene)].iloc[0]
                expr = self.scalers[0].transform([[gene_data[f'Gene1_Time_{t}'] if gene == gene_data['Gene1'] else gene_data[f'Gene2_Time_{t}']]])[0][0]
                comp = 1 if ((gene == gene_data['Gene1'] and gene_data['Gene1_Compartment'] == 'A') or
                           (gene == gene_data['Gene2'] and gene_data['Gene2_Compartment'] == 'A')) else 0
                tad = self.scalers[2].transform([[gene_data['Gene1_TAD_Boundary_Distance'] if gene == gene_data['Gene1'] else gene_data['Gene2_TAD_Boundary_Distance']]])[0][0]
                ins = self.scalers[3].transform([[gene_data['Gene1_Insulation_Score'] if gene == gene_data['Gene1'] else gene_data['Gene2_Insulation_Score']]])[0][0]
                features.append([expr, comp, tad, ins])
            self.node_features[t] = torch.tensor(features, dtype=torch.float)
    
    def create_graph(self, time_point):
        if time_point not in self.node_features:
            print(f"Time point {time_point} not found. Initializing with last known features.")
            last_known_time = max([t for t in self.node_features.keys() if t <= time_point])
            placeholder_features = self.node_features[last_known_time].clone()
            self.node_features[time_point] = placeholder_features
        
        return Data(
            x=self.node_features[time_point],
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            num_nodes=self.num_nodes
        )
    
    def get_temporal_sequences(self, future_steps=0):
        sequences, labels = [], []
        time_range = len(self.time_points)
        for i in range(time_range - self.seq_len - self.pred_len + 1):
            seq_graphs = [self.create_graph(t) for t in self.time_points[i:i + self.seq_len]]
            label_graphs = [self.create_graph(t) for t in self.time_points[i + self.seq_len:i + self.seq_len + self.pred_len]]
            sequences.append(seq_graphs)
            labels.append(label_graphs)
        
        # Add future sequences as 0.5 time steps further
        if future_steps > 0:
            future_time_points = self.time_points[-self.seq_len:]
            for _ in range(future_steps):
                future_time_points.append(future_time_points[-1] + 0.5)
                future_seq = [self.create_graph(t) for t in future_time_points[-self.seq_len:]]
                sequences.append(future_seq)
                labels.append(None)  # No ground truth for future steps (how can I control the values then ?)
        return sequences, labels

class STGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(STGCNLayer, self).__init__()
        self.temporal_conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.spatial_conv = GCNConv(out_channels, out_channels)
        self.instance_norm = nn.InstanceNorm1d(out_channels, affine=True)
        
    def forward(self, x, edge_index, edge_weight):
        x_stack = torch.stack(x).permute(1, 2, 0)  # [num_nodes, in_channels, seq_len]
        x_combined = F.relu(self.instance_norm(self.temporal_conv(x_stack)))
        x_combined = x_combined.permute(0, 2, 1)  # [num_nodes, seq_len, out_channels]
        output = [self.spatial_conv(x_combined[:, t, :], edge_index, edge_weight) for t in range(x_combined.size(1))]
        return output

class STGCN(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels, num_layers=3):
        super(STGCN, self).__init__()
        self.input_layer = STGCNLayer(in_channels, hidden_channels)
        self.hidden_layers = nn.ModuleList([STGCNLayer(hidden_channels, hidden_channels) for _ in range(num_layers - 2)])
        self.output_layer = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, graph_sequence):
        x = [g.x for g in graph_sequence]
        edge_index, edge_weight = graph_sequence[0].edge_index, graph_sequence[0].edge_attr.squeeze()
        x = self.input_layer(x, edge_index, edge_weight)
        for layer in self.hidden_layers:
            x = layer(x, edge_index, edge_weight)
        return torch.stack([self.output_layer(x_t) for x_t in x]).mean(dim=0)

def recursive_forecasting(model, initial_sequence, future_steps):
    model.eval()
    predicted_sequence = []
    input_sequence = initial_sequence.copy()
    with torch.no_grad():
        for _ in range(future_steps):
            prediction = model(input_sequence)
            predicted_graph = input_sequence[-1].clone()
            predicted_graph.x = prediction
            input_sequence.pop(0)
            input_sequence.append(predicted_graph)
            predicted_sequence.append(prediction)
    return predicted_sequence

def plot_recursive_forecasting(original_data, predictions, future_steps):
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(original_data)), original_data, label="Original Data")
    plt.plot(range(len(original_data), len(original_data) + future_steps), predictions[:, 0], 'r--', label="Predicted Future")
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.title("Recursive Forecasting of Future Data")
    plt.legend()
    plt.grid()
    #plt.show()
    plt.savefig('plottings_training_validation/future_pred.png')

if __name__ == "__main__":
    dataset = TemporalGraphDataset('mapped/enhanced_interactions.csv', seq_len=10, pred_len=1)
    sequences, _ = dataset.get_temporal_sequences(future_steps=5)
    model = STGCN(num_nodes=dataset.num_nodes, in_channels=4, hidden_channels=32, out_channels=4, num_layers=3)
    initial_sequence = sequences[-1]
    
    future_steps = 5
    predictions = recursive_forecasting(model, initial_sequence, future_steps)
    predictions = np.array(predictions)
    
    # Visualize predictions
    original_data = [g.x.mean().item() for g in initial_sequence]
    plot_recursive_forecasting(original_data, predictions, future_steps)
