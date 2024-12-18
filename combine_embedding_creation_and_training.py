import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
import seaborn as sns

def clean_gene_name(gene_name):
    """
    Some genes have descriptions side their names.
    """
    if pd.isna(gene_name):
        return gene_name
    return gene_name.split('(')[0].strip()

def create_graph_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    unique_genes = pd.concat([df['Gene1'], df['Gene2']]).unique()
    node_map = {gene: idx for idx,gene in enumerate(unique_genes)}
    num_nodes = len(node_map)

    adj_matrix = np.zeros((num_nodes, num_nodes)) # symmetric matrix
    for _, row in df.iterrows():
        i = node_map[row['Gene1']]
        j = node_map[row['Gene2']]
        adj_matrix[i,j] = row['HiC_Interaction']
        adj_matrix[j, i] = row['HiC_Interaction']
    
    G = nx.from_numpy_array(adj_matrix)
    edge_index = []
    edge_weights = []

    for u,v,d in G.edges(data=True):
        edge_index.append([u,v])
        edge_weights.append(d.get('weight', 1))

    edge_index = torch.tensor(edge_index).t().contiguous()
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)
    edge_weights = (edge_weights - edge_weights.mean()) / (edge_weights.std() + 1e-6)

    return G, edge_index , edge_weights, node_map

def create_node2vec_embeddings(graph, dimensions=128, walk_length=80, num_walks=10):

    node2vec = Node2Vec(
        graph,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=4,
        p=1,
        q=1,
        weight_key='weight'
    )
    model = node2vec.fit(window=10, min_count=1)
    embeddings = {node: model.wv[node] for node in graph.nodes()}
    return embeddings


class TemporalGraphDataset:
    def __init__(self, csv_file, node_features, edge_index, edge_attr, seq_len=10, pred_len=1):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.df = pd.read_csv(csv_file)
        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.time_points = sorted(node_features.keys())

    def create_graph(self, time_point):
        return Data(
            x=self.node_features[time_point],
            edge_index=self.edge_index,
            edge_attr=self.edge_attr
        )
    
    def get_temporal_sequences(self):
        sequences, labels = [], []
        for i in range(len(self.time_points) - self.seq_len - self.pred_len + 1):
            seq_graphs = [self.create_graph(t) for t in self.time_points[i:i + self.seq_len]]
            label_graphs = [self.create_graph(t) for t in self.time_points[i + self.seq_len:i + self.seq_len + self.pred_len]]
            sequences.append(seq_graphs)
            labels.append(label_graphs)
        return sequences, labels

class STGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(STGCNLayer, self).__init__()
        self.temporal_conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.spatial_conv = GCNConv(out_channels, out_channels)
        self.instance_norm = nn.InstanceNorm1d(out_channels, affine=True)

    def forward(self, x, edge_index, edge_weight):
        x_stack = torch.stack(x).permute(1, 2, 0)
        x_combined = F.relu(self.instance_norm(self.temporal_conv(x_stack)))
        x_combined = x_combined.permute(0, 2, 1)

        output = []
        for t in range(x_combined.size(1)):
            x_t = x_combined[:, t, :]
            edge_weight_norm = F.softmax(edge_weight, dim=0)
            out_t = self.spatial_conv(x_t, edge_index, edge_weight_norm)
            output.append(F.relu(out_t))
        return output
    
class STGCN(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels, num_layers=3):
        super(STGCN, self).__init__()
        self.input_layer = STGCNLayer(in_channels, hidden_channels)
        self.hidden_layers = nn.ModuleList([STGCNLayer(hidden_channels, hidden_channels) for _ in range(num_layers - 2)])
        self.output_layer = nn.Linear(hidden_channels, out_channels)

    def forward(self, graph_sequence):
        x = [g.x for g in graph_sequence]
        edge_index = graph_sequence[0].edge_index
        edge_weight = graph_sequence[0].edge_attr.squeeze()

        x = self.input_layer(x, edge_index, edge_weight)
        for layer in self.hidden_layers:
            x = layer(x, edge_index, edge_weight)

        return torch.stack([self.output_layer(x_t) for x_t in x]).mean(dim=0)


def train_model(model, train_sequences, train_labels, val_sequences, val_labels, num_epochs=50, learning_rate=0.0001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,  weight_decay=1e-5)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for seq, label in zip(train_sequences, train_labels):
            optimizer.zero_grad()
            output = model(seq)
            target = torch.stack([g.x for g in label]).mean(dim=0)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seq, label in zip(val_sequences, val_labels):
                output = model(seq)
                target = torch.stack([g.x for g in label]).mean(dim=0)
                val_loss += criterion(output, target).item()

        train_losses.append(total_loss / len(train_sequences))
        val_losses.append(val_loss / len(val_sequences))

        print(f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}")

    return train_losses, val_losses

def visualize_predictions_detailed(model, test_sequences, test_labels, save_dir='plottings_embedding_new'):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        all_predictions = []
        all_targets = []
        
        for seq, label in zip(test_sequences, test_labels):
            pred = model(seq)
            target = torch.stack([g.x for g in label]).mean(dim=0)
            all_predictions.append(pred)
            all_targets.append(target)
        
        predictions = torch.stack(all_predictions).numpy()
        targets = torch.stack(all_targets).numpy()
        
        # Scatter Plot: Predictions vs. Actual
        n_dims = min(predictions.shape[-1], 6)  # Visualize up to 6 dimensions
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i in range(n_dims):
            ax = axes[i]
            ax.scatter(targets[..., i].flatten(), 
                      predictions[..., i].flatten(), 
                      alpha=0.5, color='blue')
            
            min_val = min(targets[..., i].min(), predictions[..., i].min())
            max_val = max(targets[..., i].max(), predictions[..., i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
            
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'Dimension {i+1} Scatter Plot')
            ax.legend()
            
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'scatter_plots.png'))
        plt.close()
  
        # Time-Series Comparison: Actual vs. Predicted
        sample_idx = 0
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i in range(n_dims):
            ax = axes[i]
            ax.plot(targets[sample_idx, :, i], 'b-', label='Actual', marker='o')
            ax.plot(predictions[sample_idx, :, i], 'r--', label='Predicted', marker='s')
            ax.set_title(f'Time-Series Comparison (Dim {i+1})')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.legend()
            
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'time_series_comparison.png'))
        plt.close()
        
        # Histogram: Actual vs. Predicted Distributions
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i in range(n_dims):
            ax = axes[i]
            ax.hist(targets[..., i].flatten(), bins=30, alpha=0.5, label='Actual', color='blue')
            ax.hist(predictions[..., i].flatten(), bins=30, alpha=0.5, label='Predicted', color='red')
            ax.set_title(f'Distribution Comparison (Dim {i+1})')
            ax.legend()
            
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'distribution_comparison.png'))
        plt.close()
        
        # Heatmap: Prediction Errors
        errors = np.abs(predictions - targets)
        plt.figure(figsize=(10, 6))
        sns.heatmap(errors[0], cmap='YlOrRd', annot=False, fmt=".2f")
        plt.title('Prediction Error Heatmap (Sample 1)')
        plt.xlabel('Dimension')
        plt.ylabel('Time Step')
        plt.savefig(os.path.join(save_dir, 'error_heatmap.png'))
        plt.close()
 
        # Metrics: MSE, MAE, and Correlations
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        
        correlations = []
        for i in range(predictions.shape[-1]):
            corr = np.corrcoef(predictions[..., i].flatten(), 
                               targets[..., i].flatten())[0, 1]
            correlations.append(corr)
        
        metrics = {
            'MSE': mse,
            'MAE': mae,
            'Correlations': correlations
        }
        
        with open(os.path.join(save_dir, 'prediction_metrics.txt'), 'w') as f:
            f.write('Prediction Metrics:\n')
            f.write(f'MSE: {mse:.6f}\n')
            f.write(f'MAE: {mae:.6f}\n')
            f.write('\nCorrelations by dimension:\n')
            for i, corr in enumerate(correlations):
                f.write(f'Dimension {i+1}: {corr:.6f}\n')
                
        return metrics


if __name__ == "__main__":
    csv_file = 'mapped/enhanced_interactions.csv'

    G, edge_index, edge_attr, node_map = create_graph_from_csv(csv_file)

    time_points = sorted([float(col.split('_')[-1]) for col in pd.read_csv(csv_file).columns if 'Time_' in col and 'Gene1' in col])
    node_features = {}
    for t in time_points:
        print(f"Generating Node2Vec embeddings for time {t}")
        embeddings = create_node2vec_embeddings(G)
        node_features[t] = torch.tensor(list(embeddings.values()), dtype=torch.float)

    dataset = TemporalGraphDataset(csv_file, node_features, edge_index, edge_attr)
    sequences, labels = dataset.get_temporal_sequences()

    train_seq, val_seq, train_labels, val_labels = train_test_split(sequences, labels, test_size=0.2, random_state=42)

    model = STGCN(
        num_nodes=len(node_map),
        in_channels=128,
        hidden_channels=32,
        out_channels=128,
        num_layers=3
    )

    train_model(model, train_seq, train_labels, val_seq, val_labels)
    visualize_predictions_detailed(model, val_seq, val_labels)