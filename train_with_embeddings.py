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
    def __init__(self, csv_file, embedding_dir='embeddings_txt', seq_len=10, pred_len=1):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.df = pd.read_csv(csv_file)
        self.embedding_dir = embedding_dir
        self.process_data()
    
    def process_data(self):
        self.time_cols = [col for col in self.df.columns if 'Time_' in col]
        self.time_points = sorted(list(set([float(col.split('_')[-1]) 
                                          for col in self.time_cols if 'Gene1' in col])))
        unique_genes = pd.concat([self.df['Gene1'], self.df['Gene2']]).unique()
        self.node_map = {gene: idx for idx, gene in enumerate(unique_genes)}
        self.num_nodes = len(self.node_map)
        print(f"Number of nodes: {self.num_nodes}")

        self.G = self.create_graph_structure()
        self.process_node_features()
    
    def create_adjacency_matrix(self):
        adj_matrix = np.zeros((self.num_nodes, self.num_nodes)) # symmetric matrix
        for _, row in self.df.iterrows():
            i = self.node_map[row['Gene1']] 
            j = self.node_map[row['Gene2']]
            adj_matrix[i,j] = row['HiC_Interaction']
            adj_matrix[j,i] = row['HiC_Interaction']
        return adj_matrix
    
    def create_graph_structure(self):
        adj_matrix = self.create_adjacency_matrix()
        G = nx.from_numpy_array(adj_matrix)

        edge_index = []
        edge_weights = []

        for u,v,d in G.edges(data=True):
            edge_index.append([u,v])
            edge_weights.append(d['weight'])
        
        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        edge_weights = (edge_weights - edge_weights.mean()) / (edge_weights.std() + 1e-6)

        self.edge_index = edge_index
        self.edge_attr = edge_weights.unsqueeze(1)

        return G
    
    def process_node_features(self):
        self.node_features = {}
        scaler = StandardScaler()

        for t in self.time_points:
            embeddings = {}
            embedding_file = os.path.join(self.embedding_dir, f'embeddings_time_{t}.txt')
            if not os.path.exists(embedding_file):
                raise FileNotFoundError(f"No embedding file found for time {t}")

            with open(embedding_file, 'r') as f:
                for line in f:
                    if line.startswith('#'): # handle comment line for the files
                        continue
                    parts = line.strip().split()
                    gene = parts[0]
                    embedding = np.array([float(x) for x in parts[1:]])
                    embeddings[gene] = embedding
            
            features = []
            for gene in self.node_map.keys():
                if gene not in embeddings:
                    print(f"Warning: No embedding found for gene {gene} at time {t}")
                    embedding = np.zeros(128)
                else:
                    embedding = embeddings[gene]
                features.append(embedding)
            
            features = np.array(features)
            if t == self.time_points[0]:
                scaler.fit(features)
            features_normalized = scaler.transform(features)

            self.node_features[t] = torch.tensor(features_normalized, dtype=torch.float)

    def create_graph(self, time_point):
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
        
        try:
            for i in range(len(self.time_points) - self.seq_len - self.pred_len + 1):
                # Get sequence of graphs
                seq_graphs = [self.create_graph(t) for t in self.time_points[i:i+self.seq_len]]
                label_graphs = [self.create_graph(t) for t in 
                            self.time_points[i+self.seq_len:i+self.seq_len+self.pred_len]]
                
                # Debug print
                print(f"\nSequence {i}:")
                print(f"Input times: {self.time_points[i:i+self.seq_len]}")
                print(f"Target times: {self.time_points[i+self.seq_len:i+self.seq_len+self.pred_len]}")
                
                # Check for valid graphs
                if all(g is not None for g in seq_graphs) and all(g is not None for g in label_graphs):
                    sequences.append(seq_graphs)
                    labels.append(label_graphs)
                else:
                    print(f"Warning: Skipping sequence {i} due to invalid graphs")
                    
            if not sequences:
                raise ValueError("No valid sequences created")
                
            print(f"\nCreated {len(sequences)} sequences")
            print(f"Each sequence has {len(sequences[0])} time points")
            print(f"Feature dimension: {sequences[0][0].x.shape[1]}")
            
            return sequences, labels
            
        except Exception as e:
            print(f"Error in get_temporal_sequences: {str(e)}")
            print("Checking data:")
            print(f"Time points: {self.time_points}")
            print(f"Sequence length: {self.seq_len}")
            print(f"Prediction length: {self.pred_len}")
            if hasattr(self, 'node_features'):
                print(f"Node features available for times: {list(self.node_features.keys())}")
            raise
                
class STGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(STGCNLayer, self).__init__()
        self.temporal_conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.spatial_conv = GCNConv(out_channels, out_channels)
        self.instance_norm = nn.InstanceNorm1d(out_channels, affine=True)

        nn.init.xavier_uniform_(self.temporal_conv.weight, gain=0.1)
        nn.init.constant_(self.temporal_conv.bias, 0.0)
    
    def forward(self, x, edge_index, edge_weight):
        x_stack = torch.stack(x) # combin temporal with spatial data

        x_combined = x_stack.permute(1, 2, 0)
        x_combined = self.temporal_conv(x_combined)
        x_combined = torch.clamp(x_combined, min=-10, max=10)
        x_combined = self.instance_norm(x_combined)
        x_combined = F.relu(x_combined)
        x_combined = x_combined.permute(0, 2, 1)

        output = []
        for t in range(x_combined.size(1)):
            x_t = x_combined[:, t, :]
            edge_weight_norm = F.softmax(edge_weight, dim=0)
            out_t = self.spatial_conv(x_t, edge_index, edge_weight_norm)
            out_t = F.relu(out_t)
            out_t = torch.clamp(out_t, min=-10, max=10)
            output.append(out_t)
        
        return output

class STGCN(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels, num_layers=3):
        super(STGCN, self).__init__()
        
        self.num_layers = num_layers
        self.input_layer = STGCNLayer(in_channels, hidden_channels)
        
        self.hidden_layers = nn.ModuleList([
            STGCNLayer(hidden_channels, hidden_channels)
            for _ in range(num_layers-2)
        ])
        
        self.output_layer = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, graph_sequence):
        x = [g.x for g in graph_sequence]
        edge_index = graph_sequence[0].edge_index
        edge_weight = graph_sequence[0].edge_attr.squeeze()
        
        x = self.input_layer(x, edge_index, edge_weight)
        
        for layer in self.hidden_layers:
            x = layer(x, edge_index, edge_weight)
        
        outputs = []
        for x_t in x:
            out_t = self.output_layer(x_t)
            outputs.append(out_t)
        
        return torch.stack(outputs).mean(dim=0)
    


def train_model(model, train_sequences, train_labels, val_sequences, val_labels, num_epochs=100, learning_rate=0.0001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    os.makedirs('plottings_embedding', exist_ok=True)
    
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for seq,label in zip(train_sequences, train_labels):
            optimizer.zero_grad()

            output = model(seq)
            target = torch.stack([g.x for g in label]).mean(dim=0)

            if torch.isnan(output).any() or torch.isnan(target).any():
                print("Nan values detected, Skipping batch.")
                continue

            loss = criterion(output,target)

            if torch.isnan(loss):
                print(f"Nan loss detected in epoch {epoch}. Skipping batch.")
                continue

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
        
        train_loss = total_loss/len(train_sequences)
        val_loss = val_loss/len(val_sequences)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}\n')

        if epoch % 10 == 0:
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Training Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Progress with Node2Vec Embeddings')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'plottings_embedding/loss_epoch_{epoch}.png')
            plt.close()
    
    return train_losses, val_losses

def visualize_predictions_detailed(model, test_sequences, test_labels, save_dir='plottings_embedding'):

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
        
        n_dims = min(predictions.shape[-1], 6) 
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i in range(n_dims):
            ax = axes[i]
            ax.scatter(targets[..., i].flatten(), 
                      predictions[..., i].flatten(), 
                      alpha=0.5)
            
            min_val = min(targets[..., i].min(), predictions[..., i].min())
            max_val = max(targets[..., i].max(), predictions[..., i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
            
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'Dimension {i+1}')
            ax.legend()
            
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'prediction_scatter_plots.png'))
        plt.close()
  
        sample_idx = 0
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i in range(n_dims):
            ax = axes[i]
            ax.plot(targets[sample_idx, :, i], 'b-', label='Actual', marker='o')
            ax.plot(predictions[sample_idx, :, i], 'r--', label='Predicted', marker='s')
            ax.set_title(f'Dimension {i+1} Over Time')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.legend()
            
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'time_series_comparison.png'))
        plt.close()
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i in range(n_dims):
            ax = axes[i]
            ax.hist(targets[..., i].flatten(), bins=30, alpha=0.5, label='Actual', color='blue')
            ax.hist(predictions[..., i].flatten(), bins=30, alpha=0.5, label='Predicted', color='red')
            ax.set_title(f'Distribution for Dimension {i+1}')
            ax.legend()
            
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'distribution_comparison.png'))
        plt.close()
        
        errors = np.abs(predictions - targets)
        plt.figure(figsize=(10, 6))
        sns.heatmap(errors[0], cmap='YlOrRd')
        plt.title('Prediction Error Heatmap')
        plt.xlabel('Dimension')
        plt.ylabel('Time Step')
        plt.savefig(os.path.join(save_dir, 'error_heatmap.png'))
        plt.close()
 
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

    dataset = TemporalGraphDataset(
        csv_file='mapped/enhanced_interactions.csv',
        embedding_dir='embeddings_txt',
        seq_len=10,
        pred_len=1
    )

    sequences, labels = dataset.get_temporal_sequences()

    print("\nDataset information:")
    print(f"Number of sequences: {len(sequences)}")
    print(f"Sequence length: {len(sequences[0])}")
    print(f"Number of nodes: {dataset.num_nodes}")
    print(f"Feature dimension: {sequences[0][0].x.shape[1]}")

    train_seq, val_seq, train_labels, val_labels = train_test_split(
        sequences, labels, test_size=0.2, random_state=42
    )

    model = STGCN(
        num_nodes=dataset.num_nodes,
        in_channels=128,  # Node2Vec embedding dimension
        hidden_channels=64,
        out_channels=128,  # Same as input for prediction
        num_layers=3
    )

    train_losses, val_losses = train_model(
        model, train_seq, train_labels, val_seq, val_labels,
        num_epochs=100,
        learning_rate=0.001
    )

    print("\nCreating prediction visualizations...")
    metrics = visualize_predictions_detailed(model, val_seq, val_labels)
    
    print("\nPrediction Metrics:")
    print(f"MSE: {metrics['MSE']:.6f}")
    print(f"MAE: {metrics['MAE']:.6f}")
    print("\nCorrelations by dimension:")
    for i, corr in enumerate(metrics['Correlations']):
        print(f"Dimension {i+1}: {corr:.6f}")

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress with Node2Vec Embeddings')
    plt.legend()
    plt.grid(True)
    plt.savefig('plottings_embedding/final_training_progress.png')