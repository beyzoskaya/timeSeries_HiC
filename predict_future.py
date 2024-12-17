import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# Dataset Class
class TemporalGraphDataset:
    def __init__(self, csv_file, seq_len=10, pred_len=1):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.df = pd.read_csv(csv_file)
        self.process_data()
    
    def process_data(self):
        self.time_cols = [col for col in self.df.columns if 'Time_' in col]
        self.time_points = sorted(list(set([float(col.split('_')[-1]) for col in self.time_cols if 'Gene1' in col])))
        
        # Map genes to nodes
        unique_genes = pd.concat([self.df['Gene1'], self.df['Gene2']]).unique()
        self.node_map = {gene: idx for idx, gene in enumerate(unique_genes)}
        self.num_nodes = len(self.node_map)
        
        # Graph structure
        self.edge_index, self.edge_attr = self.create_graph_structure()
        
        # Node features
        self.node_features = self.process_node_features()
    
    def create_graph_structure(self):
        adj_matrix = np.zeros((self.num_nodes, self.num_nodes))
        for _, row in self.df.iterrows():
            i = self.node_map[row['Gene1']]
            j = self.node_map[row['Gene2']]
            adj_matrix[i, j] = row['HiC_Interaction']
            adj_matrix[j, i] = row['HiC_Interaction']
        edge_index = torch.tensor(np.array(np.nonzero(adj_matrix)), dtype=torch.long)
        edge_attr = torch.tensor(adj_matrix[adj_matrix.nonzero()], dtype=torch.float)
        return edge_index, edge_attr.unsqueeze(1)
    
    def process_node_features(self):
        scalers = [StandardScaler() for _ in range(4)]
        all_features = {i: [] for i in range(4)}
        
        for t in self.time_points:
            for gene in self.node_map.keys():
                gene_data = self.df[(self.df['Gene1'] == gene) | (self.df['Gene2'] == gene)].iloc[0]
                features = [
                    gene_data[f'Gene1_Time_{t}'] if gene == gene_data['Gene1'] else gene_data[f'Gene2_Time_{t}'],
                    1 if gene_data['Gene1_Compartment'] == 'A' else 0,
                    gene_data['Gene1_TAD_Boundary_Distance'],
                    gene_data['Gene1_Insulation_Score']
                ]
                for i, feat in enumerate(features):
                    all_features[i].append(feat)
        
        for i in range(4):
            scalers[i].fit(np.array(all_features[i]).reshape(-1, 1))
        
        node_features = {}
        for t in self.time_points:
            features = []
            for gene in self.node_map.keys():
                gene_data = self.df[(self.df['Gene1'] == gene) | (self.df['Gene2'] == gene)].iloc[0]
                expr = scalers[0].transform([[gene_data[f'Gene1_Time_{t}']]])[0][0]
                comp = 1 if gene_data['Gene1_Compartment'] == 'A' else 0
                tad = scalers[2].transform([[gene_data['Gene1_TAD_Boundary_Distance']]])[0][0]
                ins = scalers[3].transform([[gene_data['Gene1_Insulation_Score']]])[0][0]
                features.append([expr, comp, tad, ins])
            node_features[t] = torch.tensor(features, dtype=torch.float)
        return node_features
    
    def create_graph(self, time_point):
        return Data(
            x=self.node_features[time_point],
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            num_nodes=self.num_nodes
        )
    
    def get_temporal_sequences(self, future_steps=0):
        sequences = []
        for i in range(len(self.time_points) - self.seq_len):
            seq_graphs = [self.create_graph(t) for t in self.time_points[i:i + self.seq_len]]
            sequences.append(seq_graphs)
        
        # Future points
        future_time_points = self.time_points[-self.seq_len:]
        for _ in range(future_steps):
            future_time_points.append(future_time_points[-1] + 0.5)
            future_seq = [self.create_graph(t) for t in future_time_points[-self.seq_len:]]
            sequences.append(future_seq)
        return sequences

# Model Classes
class STGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(STGCNLayer, self).__init__()
        self.temporal_conv = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.spatial_conv = GCNConv(out_channels, out_channels)
    
    def forward(self, x, edge_index, edge_weight):
        x = torch.stack(x).permute(1, 2, 0)
        x = F.relu(self.temporal_conv(x))
        x = x.permute(0, 2, 1)
        output = [self.spatial_conv(x[:, t, :], edge_index, edge_weight) for t in range(x.size(1))]
        return output

class STGCN(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels):
        super(STGCN, self).__init__()
        self.input_layer = STGCNLayer(in_channels, hidden_channels)
        self.output_layer = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, graph_sequence):
        x = [g.x for g in graph_sequence]
        edge_index, edge_weight = graph_sequence[0].edge_index, graph_sequence[0].edge_attr.squeeze()
        x = self.input_layer(x, edge_index, edge_weight)
        return torch.stack([self.output_layer(x_t) for x_t in x]).mean(dim=0)

# Training and Forecasting
def train_model(model, train_seq, train_labels, num_epochs=100, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(num_epochs):
        model.train()
        loss = 0
        for seq, label in zip(train_seq, train_labels):
            optimizer.zero_grad()
            output = model(seq)
            target = torch.stack([g.x for g in label]).mean(dim=0)
            l = criterion(output, target)
            l.backward()
            optimizer.step()
            loss += l.item()
        print(f"Epoch {epoch+1}, Loss: {loss/len(train_seq):.4f}")

def recursive_forecasting(model, initial_sequence, future_steps):
    model.eval()
    predicted_sequence = []
    input_seq = initial_sequence.copy()
    with torch.no_grad():
        for _ in range(future_steps):
            prediction = model(input_seq)
            predicted_graph = input_seq[-1].clone()
            predicted_graph.x = prediction
            input_seq.pop(0)
            input_seq.append(predicted_graph)
            predicted_sequence.append(prediction)
    return predicted_sequence

def recursive_forecasting_with_noise(model, initial_sequence, future_steps, noise_std=0.01):
    """
    Predict future time steps recursively while adding small noise for variability.
    
    Args:
        model: Trained STGCN model.
        initial_sequence: List of input sequence.
        future_steps: Number of future steps to predict.
        noise_std: Standard deviation of Gaussian noise to introduce variability.
        
    Returns:
        List of predicted node features for each future step.
    """
    model.eval()
    predicted_sequence = []
    input_sequence = initial_sequence.copy() 

    with torch.no_grad():
        for step in range(future_steps):
            prediction = model(input_sequence)
            
            noise = torch.randn_like(prediction) * noise_std
            prediction_with_noise = prediction + noise

            predicted_graph = input_sequence[-1].clone()  
            predicted_graph.x = prediction_with_noise 
            
            input_sequence.pop(0)  
            input_sequence.append(predicted_graph) 
            
            predicted_sequence.append(prediction_with_noise)

    return predicted_sequence

def plot_recursive_forecasting(original_data, predictions, future_steps, title="Future Predictions"):
    
    plt.figure(figsize=(12, 6))
    
    # Plot original data
    plt.plot(range(len(original_data)), original_data, label="Original Data", marker='o')
    
    # Plot predictions
    future_indices = range(len(original_data), len(original_data) + future_steps)
    predicted_means = [p.mean().item() for p in predictions]
    plt.plot(future_indices, predicted_means, 'r--', label="Predicted Future", marker='x')
    
    plt.xlabel("Time Steps")
    plt.ylabel("Mean Node Features")
    plt.title(title)
    plt.legend()
    plt.grid()
    #plt.show()
    plt.savefig('plottings_training_validation/original_vs_predicted_recursive_training_with_noise.png')


# Main Script
if __name__ == "__main__":
    dataset = TemporalGraphDataset('mapped/enhanced_interactions.csv', seq_len=10)
    sequences = dataset.get_temporal_sequences()
    
    train_seq, val_seq = train_test_split(sequences, test_size=0.2, random_state=42)
    
    model = STGCN(num_nodes=dataset.num_nodes, in_channels=4, hidden_channels=32, out_channels=4)
    train_model(model, train_seq, val_seq)
    
    future_steps = 5
    initial_sequence = sequences[-1]  # Start with the last sequence
    predictions = recursive_forecasting_with_noise(model, initial_sequence, future_steps)
    
    original_data = [g.x.mean().item() for g in initial_sequence]
    plot_recursive_forecasting(original_data, predictions, future_steps, title="Recursive Forecasting with Noise")

