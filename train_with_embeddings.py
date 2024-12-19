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
        print(f"Found {len(self.time_points)} time points")
        
        unique_genes = pd.concat([self.df['Gene1'], self.df['Gene2']]).unique()
        self.node_map = {gene: idx for idx, gene in enumerate(unique_genes)}
        self.num_nodes = len(self.node_map)
        print(f"Number of nodes: {self.num_nodes}")
        
        self.create_graph_structure()
        self.load_node_embeddings()
    
    def create_graph_structure(self):
        """Create graph structure using HiC interactions"""
        adj_matrix = np.zeros((self.num_nodes, self.num_nodes))
        for _, row in self.df.iterrows():
            i = self.node_map[row['Gene1']]
            j = self.node_map[row['Gene2']]
            adj_matrix[i, j] = row['HiC_Interaction']
            adj_matrix[j, i] = row['HiC_Interaction']
        
        edge_index = []
        edge_weights = []
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):
                if adj_matrix[i, j] > 0:
                    edge_index.extend([[i, j], [j, i]])
                    edge_weights.extend([adj_matrix[i, j], adj_matrix[i, j]])
        
        self.edge_index = torch.tensor(edge_index).t().contiguous()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        
        # Normalize edge weights
        edge_weights = (edge_weights - edge_weights.mean()) / (edge_weights.std() + 1e-6)
        self.edge_attr = edge_weights.unsqueeze(1)
        
        print(f"Graph structure created: {len(edge_weights)} edges")
    
    def load_node_embeddings(self):
        # Standard Scaler is removed for node embedding loading because there is a mismatch
        self.node_features = {}
        
        for t in self.time_points:
            embedding_file = os.path.join(self.embedding_dir, f'embeddings_time_{t}.txt')
            if not os.path.exists(embedding_file):
                raise FileNotFoundError(f"No embedding file found for time {t}")
            
            embeddings = {}
            with open(embedding_file, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split()
                    gene = parts[0]
                    embedding = np.array([float(x) for x in parts[1:]])
                    embeddings[gene] = embedding
            
            features = []
            for gene in self.node_map.keys():
                if gene in embeddings:
                    features.append(embeddings[gene])
                else:
                    print(f"Warning: No embedding for gene {gene} at time {t}")
                    features.append(np.zeros(128))
            
            # Store features without additional normalization
            self.node_features[t] = torch.tensor(np.array(features), dtype=torch.float)
            
            if t == self.time_points[0]:
                print(f"Feature dimension: {self.node_features[t].shape[1]}")
                print(f"Feature range: [{self.node_features[t].min():.4f}, {self.node_features[t].max():.4f}]")
    
    def create_graph(self, time_point):
        """Create graph with embeddings as node features"""
        return Data(
            x=self.node_features[time_point],
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            num_nodes=self.num_nodes
        )
    
    def get_temporal_sequences(self):
        """Create sequences of temporal graphs for training"""
        sequences = []
        labels = []
        
        for i in range(len(self.time_points) - self.seq_len - self.pred_len + 1):
            seq_graphs = [self.create_graph(t) for t in self.time_points[i:i+self.seq_len]]
            label_graphs = [self.create_graph(t) for t in 
                          self.time_points[i+self.seq_len:i+self.seq_len+self.pred_len]]
            
            if i == 0:
                print("\nSequence information:")
                print(f"Input time points: {self.time_points[i:i+self.seq_len]}")
                print(f"Target time points: {self.time_points[i+self.seq_len:i+self.seq_len+self.pred_len]}")
                print(f"Feature dimension: {seq_graphs[0].x.shape[1]}")
                print(f"Feature range: [{seq_graphs[0].x.min():.4f}, {seq_graphs[0].x.max():.4f}]")
            
            sequences.append(seq_graphs)
            labels.append(label_graphs)
        
        print(f"\nCreated {len(sequences)} sequences")
        return sequences, labels
                
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
    


def train_model(model, train_sequences, train_labels, val_sequences, val_labels, num_epochs=100, learning_rate=0.0005):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    os.makedirs('plottings_embedding_combined', exist_ok=True)
    
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

def visualize_predictions_detailed(model, test_sequences, test_labels, save_dir='plottings_embedding_combined'):

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


def calculate_prediction_metrics(predictions, targets, dim_names=None):
    
    if dim_names is None:
        dim_names = [f"Dimension_{i}" for i in range(predictions.shape[-1])]
    
    metrics = {}
    
    for dim in range(predictions.shape[-1]):
        pred = predictions[..., dim].flatten()
        true = targets[..., dim].flatten()
        
        mse = np.mean((pred - true) ** 2)
        mae = np.mean(np.abs(pred - true))
        
        corr = np.corrcoef(pred, true)[0, 1]
        
        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        metrics[dim_names[dim]] = {
            'MSE': mse,
            'MAE': mae,
            'Correlation': corr,
            'R2': r2
        }

    metrics['Overall'] = {
        'MSE': np.mean((predictions - targets) ** 2),
        'MAE': np.mean(np.abs(predictions - targets)),
        'Correlation': np.mean([m['Correlation'] for m in metrics.values() if isinstance(m, dict)]),
        'R2': np.mean([m['R2'] for m in metrics.values() if isinstance(m, dict)])
    }
    
    return metrics

def plot_prediction_accuracy(predictions, targets, save_dir='plottings_embedding_combined'):
   
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Overall scatter plot
    plt.figure(figsize=(10, 10))
    plt.scatter(targets.flatten(), predictions.flatten(), alpha=0.1)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values (All Dimensions)')
    plt.savefig(os.path.join(save_dir, 'overall_predictions.png'))
    plt.close()
    
    # 2. Dimension-wise plots
    n_dims = min(6, predictions.shape[-1])  # Show up to 6 dimensions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i in range(n_dims):
        ax = axes[i]
        ax.scatter(targets[..., i].flatten(), 
                  predictions[..., i].flatten(), 
                  alpha=0.1)
        ax.plot([targets[..., i].min(), targets[..., i].max()], 
                [targets[..., i].min(), targets[..., i].max()], 
                'r--')
        ax.set_title(f'Dimension {i+1}')
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'dimension_predictions.png'))
    plt.close()
    
    # 3. Error distribution
    errors = predictions - targets
    plt.figure(figsize=(10, 6))
    plt.hist(errors.flatten(), bins=50)
    plt.title('Prediction Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(save_dir, 'error_distribution.png'))
    plt.close()
    
    # 4. Error heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(np.abs(errors[0]), cmap='YlOrRd')
    plt.title('Absolute Prediction Error Heatmap')
    plt.xlabel('Dimension')
    plt.ylabel('Sample')
    plt.savefig(os.path.join(save_dir, 'error_heatmap.png'))
    plt.close()

def evaluate_model(model, test_sequences, test_labels, save_dir='plottings_embedding_combined'):
   
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for seq, label in zip(test_sequences, test_labels):
            # Get predictions
            pred = model(seq)
            target = torch.stack([g.x for g in label]).mean(dim=0)
            
            all_predictions.append(pred.numpy())
            all_targets.append(target.numpy())
    
    predictions = np.stack(all_predictions)
    targets = np.stack(all_targets)
    
    metrics = calculate_prediction_metrics(predictions, targets)
    
    plot_prediction_accuracy(predictions, targets, save_dir)
    
    with open(os.path.join(save_dir, 'prediction_metrics.txt'), 'w') as f:
        f.write('Prediction Metrics:\n\n')
        for dim, dim_metrics in metrics.items():
            f.write(f'\n{dim}:\n')
            for metric_name, value in dim_metrics.items():
                f.write(f'{metric_name}: {value:.4f}\n')
    
    return metrics

if __name__ == "__main__":
    # Load and prepare data
    dataset = TemporalGraphDataset(
        csv_file='mapped/enhanced_interactions.csv',
        embedding_dir='embeddings_txt',
        seq_len=10,
        pred_len=1
    )
    
    sequences, labels = dataset.get_temporal_sequences()
    
    # Split data
    train_seq, val_seq, train_labels, val_labels = train_test_split(
        sequences, labels, test_size=0.2, random_state=42
    )
    
    # Create model
    model = STGCN(
        num_nodes=dataset.num_nodes,
        in_channels=128,  # Node2Vec embedding dimension
        hidden_channels=32,
        out_channels=128,
        num_layers=3
    )
    
    # Train model
    train_losses, val_losses = train_model(
        model, train_seq, train_labels, val_seq, val_labels
    )

    
    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress with Node2Vec Embeddings')
    plt.legend()
    plt.grid(True)
    plt.savefig('plottings_embedding_combined/training_progress.png')

    print("\nEvaluating model predictions...")
    metrics = evaluate_model(model, val_seq, val_labels)
    
    print("\nOverall Prediction Metrics:")
    for metric_name, value in metrics['Overall'].items():
        print(f"{metric_name}: {value:.4f}")
    
    
   