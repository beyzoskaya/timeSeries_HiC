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
        #print(f"edge index: {edge_index}") --> not nan
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        #print(f"edge weights: {edge_weights}") --> not nan
        
        # Normalize edge weights
        edge_weights = (edge_weights - edge_weights.mean()) / (edge_weights.std() + 1e-6)
        #print(f"Normalized edge weights: {edge_weights}") --> not nan
        
        self.edge_index = edge_index
        self.edge_attr = edge_weights.unsqueeze(1)
        
        #print("Graph structure created:")
        #print(f"Adjacency matrix shape: {adj_matrix.shape}")
        #print(f"Edge index shape: {self.edge_index.shape}")
        #print(f"Edge weights shape: {self.edge_attr.shape}")
        
        if torch.isnan(self.edge_index).any():
            print("NaN found in edge index")
        if torch.isnan(self.edge_attr).any():
            print("NaN found in edge weights")
        if (self.edge_attr == 0).all():
            print("Warning: All edge weights are zero")
        
        return G
    
    def process_node_features(self):
        """Process node features with normalization"""
        self.node_features = {}
        
        # Create scalers for each feature type
        self.scalers = [StandardScaler() for _ in range(4)]
        
        # Collect all features for fitting scalers
        all_features = {i: [] for i in range(4)}
        for t in self.time_points:
            for gene in self.node_map.keys():
                gene_data = self.df[(self.df['Gene1'] == gene) | 
                                  (self.df['Gene2'] == gene)].iloc[0]
                #print(f"For {gene} gene data is: {gene_data}")
                features = [
                    gene_data[f'Gene1_Time_{t}'] if gene == gene_data['Gene1'] 
                    else gene_data[f'Gene2_Time_{t}'],
                    1 if ((gene == gene_data['Gene1'] and gene_data['Gene1_Compartment'] == 'A') or
                         (gene == gene_data['Gene2'] and gene_data['Gene2_Compartment'] == 'A')) else 0,
                    gene_data['Gene1_TAD_Boundary_Distance'] if gene == gene_data['Gene1']
                    else gene_data['Gene2_TAD_Boundary_Distance'],
                    gene_data['Gene1_Insulation_Score'] if gene == gene_data['Gene1']
                    else gene_data['Gene2_Insulation_Score']
                ]
                
                for i, feat in enumerate(features):
                    all_features[i].append(feat)
        
        # Fit scalers
        for i in range(4):
            if i != 1:  # Don't normalize binary compartment feature
                self.scalers[i].fit(np.array(all_features[i]).reshape(-1, 1))
        
        # Transform features
        for t in self.time_points:
            features = []
            for gene in self.node_map.keys():
                gene_data = self.df[(self.df['Gene1'] == gene) | 
                                  (self.df['Gene2'] == gene)].iloc[0]
                
                # Get and normalize features
                expr = self.scalers[0].transform([[gene_data[f'Gene1_Time_{t}'] if gene == gene_data['Gene1']
                        else gene_data[f'Gene2_Time_{t}']]])[0][0]
                comp = 1 if ((gene == gene_data['Gene1'] and gene_data['Gene1_Compartment'] == 'A') or
                           (gene == gene_data['Gene2'] and gene_data['Gene2_Compartment'] == 'A')) else 0
                tad = self.scalers[2].transform([[gene_data['Gene1_TAD_Boundary_Distance'] if gene == gene_data['Gene1']
                        else gene_data['Gene2_TAD_Boundary_Distance']]])[0][0]
                ins = self.scalers[3].transform([[gene_data['Gene1_Insulation_Score'] if gene == gene_data['Gene1']
                        else gene_data['Gene2_Insulation_Score']]])[0][0]
                
                features.append([expr, comp, tad, ins])
            
            self.node_features[t] = torch.tensor(features, dtype=torch.float)
            #print(f"Features for the nodes: {self.node_features}")
    
    def create_graph(self, time_point):
        return Data(
            x=self.node_features[time_point],
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            num_nodes=self.num_nodes
        )
    
    def get_temporal_sequences(self):
        """Debug temporal sequence creation"""
        sequences = []
        labels = []
        
        for i in range(len(self.time_points) - self.seq_len - self.pred_len + 1):
            # Print time points being used
            #print(f"\nSequence {i}:")
            #print(f"Input times: {self.time_points[i:i+self.seq_len]}")
            #print(f"Target times: {self.time_points[i+self.seq_len:i+self.seq_len+self.pred_len]}")
            
            seq_graphs = [self.create_graph(t) for t in self.time_points[i:i+self.seq_len]]
            label_graphs = [self.create_graph(t) for t in 
                        self.time_points[i+self.seq_len:i+self.seq_len+self.pred_len]]
            
            # Check graph structures
            print("\nInput graphs:")
            for j, g in enumerate(seq_graphs):
                print(f"Time {j}: nodes={g.x.shape}, edges={g.edge_index.shape}")
            
            sequences.append(seq_graphs)
            labels.append(label_graphs)
        
        return sequences, labels
        

    
class STGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        get sequence of node (genes from all chromosomes) features
        apply temporal conv to capture patterns over time
        apply graph conv to capture spatial relations
        """
        super(STGCNLayer, self).__init__()
        self.temporal_conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.spatial_conv = GCNConv(out_channels, out_channels)
        
        # for small batches because I got nan values for x at some point
        self.instance_norm = nn.InstanceNorm1d(out_channels, affine=True)
        
        nn.init.xavier_uniform_(self.temporal_conv.weight, gain=0.1)
        nn.init.constant_(self.temporal_conv.bias, 0.0)
        
    def forward(self, x, edge_index, edge_weight):
        x_stack = torch.stack(x)
        print(f"Input values - min: {x_stack.min():.4f}, max: {x_stack.max():.4f}")
        
        x_combined = x_stack.permute(1, 2, 0)  # [num_nodes, features, seq_len]
        
        # Apply temporal convolution with gradient clipping
        x_combined = self.temporal_conv(x_combined)
        x_combined = torch.clamp(x_combined, min=-10, max=10)
        
        # Apply instance normalization
        x_combined = self.instance_norm(x_combined)
        
        # ReLU activation
        x_combined = F.relu(x_combined)
        
        # Prepare for spatial conv
        x_combined = x_combined.permute(0, 2, 1)  # [num_nodes, seq_len, features]
        
        # Process each time step
        output = []
        for t in range(x_combined.size(1)):
            x_t = x_combined[:, t, :]
            
            # Normalize edge weights
            edge_weight_norm = F.softmax(edge_weight, dim=0)
            
            # Apply GCN with normalized weights
            out_t = self.spatial_conv(x_t, edge_index, edge_weight_norm)
            
            # ReLU and gradient clipping
            out_t = F.relu(out_t)
            out_t = torch.clamp(out_t, min=-10, max=10)
            
            output.append(out_t)
            
        # Print output values
        out_stack = torch.stack(output)
        print(f"Output values - min: {out_stack.min():.4f}, max: {out_stack.max():.4f}")
        
        return output
class STGCN(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels, num_layers=3):
        """
        input as sequence of graphs (10 time points because sequence lenght is 10)

        """
        super(STGCN, self).__init__()
        
        self.num_layers = num_layers
        self.input_layer = STGCNLayer(in_channels, hidden_channels)
        
        self.hidden_layers = nn.ModuleList([
            STGCNLayer(hidden_channels, hidden_channels)
            for _ in range(num_layers-2)
        ])
        
        self.output_layer = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, graph_sequence):
        # Extract features and structure
        x = [g.x for g in graph_sequence]
        edge_index = graph_sequence[0].edge_index
        edge_weight = graph_sequence[0].edge_attr.squeeze()

        print(f"Input to STGCN: min={x[0].min().item()}, max={x[0].max().item()}, mean={x[0].mean().item()}")
        
        # Process through layers
        x = self.input_layer(x, edge_index, edge_weight)
        
        #for layer in self.hidden_layers:
        for layer_idx, layer in enumerate(self.hidden_layers):
            x = layer(x, edge_index, edge_weight)
            print(f"After layer {layer_idx}: min={x[0].min().item()}, max={x[0].max().item()}, mean={x[0].mean().item()}")

        
        # Apply output layer to each time step
        outputs = []
        for x_t in x:
            out_t = self.output_layer(x_t)
            outputs.append(out_t)
        
        final_output = torch.stack(outputs).mean(dim=0)
        print(f"Final output: min={final_output.min().item()}, max={final_output.max().item()}, mean={final_output.mean().item()}")
        
        return torch.stack(outputs).mean(dim=0)
    
def evaluate_performance(model, test_sequences, test_labels):
    model.eval()
    all_predictions = []
    all_targets = []
    metrics = {}
    
    with torch.no_grad():
        for seq, label in zip(test_sequences, test_labels):
            # Get predictions
            pred = model(seq)
            target = torch.stack([g.x for g in label]).mean(dim=0)
            
            all_predictions.append(pred)
            all_targets.append(target)
        
        # Convert to tensors
        predictions = torch.stack(all_predictions)
        targets = torch.stack(all_targets)
        
        # Calculate metrics
        mse = F.mse_loss(predictions, targets).item()
        mae = F.l1_loss(predictions, targets).item()
        
        # Calculate correlation per feature
        correlations = []
        for i in range(predictions.shape[-1]):
            corr = np.corrcoef(predictions[..., i].numpy().flatten(), 
                             targets[..., i].numpy().flatten())[0,1]
            correlations.append(corr)
        
        metrics['MSE'] = mse
        metrics['MAE'] = mae
        metrics['Correlations'] = correlations
        
    return metrics, predictions, targets   

def visualize_predictions(predictions, targets, feature_idx=0):
    """
    Visualize predictions vs actual values for a specific feature
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(targets[..., feature_idx].numpy(), 
               predictions[..., feature_idx].numpy(), 
               alpha=0.5)
    plt.plot([targets[..., feature_idx].min(), targets[..., feature_idx].max()], 
             [targets[..., feature_idx].min(), targets[..., feature_idx].max()], 
             'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predictions vs Actual Values for Feature {feature_idx}')
    #plt.show()
    plt.savefig('plottings/pred_vs_actual_base_model.png')

def plot_time_series_prediction(predictions, targets, sequence_idx=0, feature_idx=0):
    """
    Plot time series of predictions vs actual values
    """
    plt.figure(figsize=(12, 6))
    plt.plot(targets[sequence_idx, :, feature_idx].numpy(), 
             label='Actual', marker='o')
    plt.plot(predictions[sequence_idx, :, feature_idx].numpy(), 
             label='Predicted', marker='o')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.title(f'Time Series Prediction vs Actual (Feature {feature_idx})')
    plt.legend()
    #plt.show()
    plt.savefig('plottings/time_series_predictions_vs_actual.png')

def evaluate_performance_save_plots(model, test_sequences, test_labels, epoch=None, save_dir='plottings'):

    model.eval()
    all_predictions = []
    all_targets = []
    metrics = {}
    
    with torch.no_grad():
        for seq, label in zip(test_sequences, test_labels):
            pred = model(seq)
            target = torch.stack([g.x for g in label]).mean(dim=0)
            all_predictions.append(pred)
            all_targets.append(target)
        
        predictions = torch.stack(all_predictions)
        targets = torch.stack(all_targets)
        
        # Calculate metrics
        mse = F.mse_loss(predictions, targets).item()
        mae = F.l1_loss(predictions, targets).item()
        
        correlations = []
        for i in range(predictions.shape[-1]):
            corr = np.corrcoef(predictions[..., i].numpy().flatten(), 
                             targets[..., i].numpy().flatten())[0,1]
            correlations.append(corr)
        
        metrics['MSE'] = mse
        metrics['MAE'] = mae
        metrics['Correlations'] = correlations

        # Save visualizations if epoch is provided
        if epoch is not None and epoch % 10 == 0:
            # Create directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)
            
            # Save predictions vs actual plot for each feature
            for feature_idx in range(predictions.shape[-1]):
                plt.figure(figsize=(10, 6))
                plt.scatter(targets[..., feature_idx].numpy(), 
                          predictions[..., feature_idx].numpy(), 
                          alpha=0.5)
                plt.plot([targets[..., feature_idx].min(), targets[..., feature_idx].max()], 
                        [targets[..., feature_idx].min(), targets[..., feature_idx].max()], 
                        'r--')
                plt.xlabel('Actual Values')
                plt.ylabel('Predicted Values')
                plt.title(f'Feature {feature_idx} Predictions (Epoch {epoch})')
                plt.savefig(f'{save_dir}/pred_vs_actual_feature{feature_idx}_epoch{epoch}.png')
                plt.close()

            # Save time series plot
            plt.figure(figsize=(12, 6))
            for feature_idx in range(predictions.shape[-1]):
                plt.subplot(2, 2, feature_idx + 1)
                plt.plot(targets[0, :, feature_idx].numpy(), 
                        label='Actual', marker='o')
                plt.plot(predictions[0, :, feature_idx].numpy(), 
                        label='Predicted', marker='o')
                plt.title(f'Feature {feature_idx}')
                plt.legend()
            plt.tight_layout()
            plt.savefig(f'{save_dir}/time_series_epoch{epoch}.png')
            plt.close()

    return metrics, predictions, targets



def train_model(model, train_sequences, train_labels, val_sequences, val_labels, 
                num_epochs=100, learning_rate=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()

    os.makedirs('plottings_training_validation', exist_ok=True)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for seq, label in zip(train_sequences, train_labels):
            optimizer.zero_grad()
            
            # Forward pass
            output = model(seq)
            target = torch.stack([g.x for g in label]).mean(dim=0)
            
            # Debug prints
            if epoch % 10 == 0:
                metrics, _ , _ = evaluate_performance_save_plots(model, val_sequences, val_labels, 
                                              epoch=epoch, save_dir='plottings')
                print("\nValidation Metrics:")
                print(f"MSE: {metrics['MSE']:.4f}")
                print(f"MAE: {metrics['MAE']:.4f}")
                print("Feature Correlations:", 
                    [f"{corr:.4f}" for corr in metrics['Correlations']])
                #print(f"Input stats: mean={torch.stack([g.x for g in seq]).mean():.4f}, std={torch.stack([g.x for g in seq]).std():.4f}")
                #print(f"Output stats: mean={output.mean():.4f}, std={output.std():.4f}")
                #print(f"Target stats: mean={target.mean():.4f}, std={target.std():.4f}")
            
            # Check for NaN
            if torch.isnan(output).any() or torch.isnan(target).any():
                print("NaN values detected in output or target. Skipping this batch.")
                continue
            
            loss = criterion(output, target)
            if torch.isnan(loss):
                print(f"NaN loss detected in epoch {epoch}. Skipping this batch.")
                continue

            # Backward pass with gradient clipping
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seq, label in zip(val_sequences, val_labels):
                output = model(seq)
                target = torch.stack([g.x for g in label]).mean(dim=0)
                if not (torch.isnan(output).any() or torch.isnan(target).any()):
                    val_loss += criterion(output, target).item()
        
        train_loss = total_loss/len(train_sequences)
        val_loss = val_loss/len(val_sequences)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}\n')
        
        # Early stopping if loss is NaN
        if np.isnan(train_loss) or np.isnan(val_loss):
            print("Training stopped due to NaN loss")
            break
    
    return train_losses, val_losses

if __name__ == "__main__":
    # Load and prepare data
    dataset = TemporalGraphDataset('mapped/enhanced_interactions.csv', seq_len=10, pred_len=1)
    sequences, labels = dataset.get_temporal_sequences()

    for i, seq in enumerate(sequences):
        for graph in seq:
            if torch.isnan(graph.x).any():
                print(f"NaN found in input node features at sequence {i}")
            if torch.isinf(graph.x).any():
                print(f"Inf found in input node features at sequence {i}")

   
    
    # Print dataset information
    print("\nDataset information:")
    print(f"Number of sequences: {len(sequences)}")
    print(f"Sequence length: {len(sequences[0])}")
    print(f"Number of nodes: {dataset.num_nodes}")
    
    # Print sample graph information
    sample_graph = sequences[0][0]
    print("\nSample graph structure:")
    print(f"Node features: {sample_graph.x.shape}")
    print(f"Edge index: {sample_graph.edge_index.shape}")
    print(f"Edge weights: {sample_graph.edge_attr.shape}")
    
    # Split data
    train_seq, val_seq, train_labels, val_labels = train_test_split(
        sequences, labels, test_size=0.2, random_state=42)
    
    # Create model
    model = STGCN(
        num_nodes=dataset.num_nodes,
        in_channels=4,
        hidden_channels=32,  # Reduced from 64 to 32 for stability
        out_channels=4,
        num_layers=3
    )
    
    # Train model
    train_losses, val_losses = train_model(model, train_seq, train_labels, val_seq, val_labels)
    
    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig('plottings/train_val_loss_base_model.png')
