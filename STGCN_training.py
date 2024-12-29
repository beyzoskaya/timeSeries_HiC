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
sys.path.append('./STGCN')
from STGCN.model.models import STGCNChebGraphConv
import argparse

def clean_gene_name(gene_name):
    """Clean gene name by removing descriptions and extra information"""
    if pd.isna(gene_name):
        return gene_name
    return gene_name.split('(')[0].strip()

################# Analyze Data ######################
def analyze_label_structure(val_sequences, val_labels):
    """Analyze the structure of sequences and labels"""
    print("\n=== Label Structure Analysis ===")
    
    # Look at first sequence and label
    seq = val_sequences[0]
    label = val_labels[0]
    
    print("\nSequence structure:")
    print(f"Number of graphs in sequence: {len(seq)}")
    print(f"First graph features shape: {seq[0].x.shape}")
    print(f"Edge index shape: {seq[0].edge_index.shape}")
    
    print("\nLabel structure:")
    print(f"Number of graphs in label: {len(label)}")
    print(f"Label features shape: {label[0].x.shape}")
    
    # Look at actual values
    print("\nExample values for first node:")
    print("Sequence feature values:")
    for i, g in enumerate(seq):
        print(f"Time step {i}: {g.x[0, :5]}")  # First node, first 5 features
    
    print("\nLabel feature values:")
    print(f"Target: {label[0].x[0, :5]}")  # First node, first 5 features
    
    return seq[0].x.shape, label[0].x.shape

def get_raw_predictions(model, val_sequences, val_labels):
    """Get predictions without any averaging"""
    model.eval()
    with torch.no_grad():
        # Get one sequence prediction
        seq = val_sequences[0]
        label = val_labels[0]
        
        # Get prediction
        pred = model(seq)
        target = label[0].x  # Direct access without mean
        
        print("\n=== Prediction Analysis ===")
        print(f"Raw prediction shape: {pred.shape}")
        print(f"Raw target shape: {target.shape}")
        
        print("\nFirst node values:")
        print(f"Predicted: {pred[0, :5]}")  # First node, first 5 features
        print(f"Target: {target[0, :5]}")   # First node, first 5 features
        
        return pred, target

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.6, beta=0.4):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
        self.huber = nn.HuberLoss(delta=1.0)
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        huber_loss = self.huber(pred, target)
        
        # Correlation loss to preserve patterns
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        vx = pred_flat - torch.mean(pred_flat)
        vy = target_flat - torch.mean(target_flat)
        
        corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-12)
        corr_loss = 1 - corr
        
        return self.alpha * (0.5 * mse_loss + 0.5 * huber_loss) + self.beta * corr_loss

"""
number of nodes in the graph. 52
43 time points
"""
class TemporalGraphDataset:
    def __init__(self, csv_file, embedding_dim=64, seq_len=5, pred_len=1): # I change the seq_len to more lower value
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.embedding_dim = embedding_dim
        
        self.df = pd.read_csv(csv_file)
        self.df['Gene1_clean'] = self.df['Gene1'].apply(clean_gene_name)
        self.df['Gene2_clean'] = self.df['Gene2'].apply(clean_gene_name)
        
        # Create static node mapping
        unique_genes = pd.concat([self.df['Gene1_clean'], self.df['Gene2_clean']]).unique()
        self.node_map = {gene: idx for idx, gene in enumerate(unique_genes)}
        self.num_nodes = len(self.node_map)
        print(f"Number of nodes: {self.num_nodes}")
        
        # Get time points
        self.time_cols = [col for col in self.df.columns if 'Time_' in col]
        self.time_points = sorted(list(set([float(col.split('_')[-1]) 
                                          for col in self.time_cols if 'Gene1' in col])))
        print(f"Found {len(self.time_points)} time points")
        
        # Create base graph and features
        self.base_graph = self.create_base_graph()
        print("Base graph created")
        self.node_features = self.create_temporal_node_features_several_graphs_created() # try with several graphs for time series consistency
        print("Temporal node features created")
        
        # Get edge information
        self.edge_index, self.edge_attr = self.get_edge_index_and_attr()
        print(f"Graph structure created with {len(self.edge_attr)} edges")
    
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
    
    
    def create_temporal_node_features_several_graphs_created(self):
        """Create temporal node features with correct dimensionality"""
        temporal_features = {}
        
        for t in self.time_points:
            print(f"\nProcessing time point {t}")
            
            # Create graph for this time point
            G = nx.Graph()
            G.add_nodes_from(self.node_map.keys())
            
            # Add edges with weights
            for _, row in self.df.iterrows():
                gene1_expr = row[f'Gene1_Time_{t}']
                gene2_expr = row[f'Gene2_Time_{t}']
                
                # Calculate edge weights incorporating temporal information
                hic_weight = row['HiC_Interaction']
                compartment_sim = 1 if row['Gene1_Compartment'] == row['Gene2_Compartment'] else 0
                tad_dist = abs(row['Gene1_TAD_Boundary_Distance'] - row['Gene2_TAD_Boundary_Distance'])
                tad_sim = 1 / (1 + tad_dist)
                ins_sim = 1 / (1 + abs(row['Gene1_Insulation_Score'] - row['Gene2_Insulation_Score']))
                expr_sim = 1 / (1 + abs(gene1_expr - gene2_expr))
                
                # more temporal focus weights for new model
                weight = (hic_weight * 0.25 +  # 25% HiC
                compartment_sim * 0.1 +        # 10% Compartment
                tad_sim * 0.1 +                # 10% TAD
                ins_sim * 0.05 +               # 5% Insulation
                expr_sim * 0.5)
                        
                G.add_edge(row['Gene1_clean'], row['Gene2_clean'], weight=weight)
            
            # Create Node2Vec embeddings with exact embedding_dim dimensions
            node2vec = Node2Vec(
                G,
                dimensions=self.embedding_dim,  # Use full embedding dimension
                walk_length=10,
                num_walks=80,
                workers=1,
                p=1,
                q=1,
                weight_key='weight'
            )
            
            model = node2vec.fit(window=10, min_count=1)
            
            # Create features for all nodes
            features = []
            for gene in self.node_map.keys():
                # Get Node2Vec embedding
                node_embedding = torch.tensor(model.wv[gene], dtype=torch.float32)
                
                # Get expression value for this time point
                gene1_expr = self.df[self.df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
                gene2_expr = self.df[self.df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
                expr_value = gene1_expr[0] if len(gene1_expr) > 0 else \
                            (gene2_expr[0] if len(gene2_expr) > 0 else 0.0)
                
                # Modify last dimension of embedding to incorporate expression
                node_embedding[-1] = expr_value
                
                features.append(node_embedding)
            
            # Stack features
            temporal_features[t] = torch.stack(features)
            
            # Verify dimensions for first time point
            if t == self.time_points[0]:
                print(f"\nFeature verification for time {t}:")
                print(f"Feature shape: {temporal_features[t].shape}")
                print(f"Expected shape: ({len(self.node_map)}, {self.embedding_dim})")
                
                # Verify content
                first_gene = list(self.node_map.keys())[0]
                print(f"\nExample features for {first_gene}:")
                print(f"Shape: {temporal_features[t][0].shape}")
                print(f"First 5 values: {temporal_features[t][0, :5]}")
                print(f"Expression value (last dim): {temporal_features[t][0, -1]}")
        
        return temporal_features

    def get_edge_index_and_attr(self):
        """Convert base graph to PyG format"""
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
        """Create PyG graph for a specific time point"""
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

        print("\nTime points being used:")
        print(f"All time points: {self.time_points}")
        
        for i in range(len(self.time_points) - self.seq_len - self.pred_len + 1):
            seq_graphs = [self.get_pyg_graph(t) for t in self.time_points[i:i+self.seq_len]]
            label_graphs = [self.get_pyg_graph(t) for t in 
                          self.time_points[i+self.seq_len:i+self.seq_len+self.pred_len]]
            
            #input_times = self.time_points[i:i+self.seq_len]
            target_times = self.time_points[i+self.seq_len:i+self.seq_len+self.pred_len]
            
            if i == 0:
                print("\nSequence information:")
                print(f"Input time points: {self.time_points[i:i+self.seq_len]}")
                print(f"Target time points: {self.time_points[i+self.seq_len:i+self.seq_len+self.pred_len]}")
                print(f"Feature dimension: {seq_graphs[0].x.shape[1]}")

                # debugging for gene values
                #label_tensor = torch.stack([g.x for g in label_graphs]).mean(dim=0)
                label_tensor = torch.stack([g.x for g in label_graphs]).squeeze(dim=0) # Instead of mean, I directly squeeze the dim
                #print(f" Label tensor: {label_tensor}")
                #print(f" Label tensor shape: {label_tensor.shape}") # [1, 52, 32] without mean(dim=0)--> with dim=0 [52, 32] 
                genes = list(self.node_map.keys())
                print("\nSample label values for first 5 genes:")
                for idx in range(min(5, len(genes))):
                    gene = genes[idx]
                    value = label_tensor[idx]
                    print(f"{gene}: {value.mean().item():.4f}")
                
                # Check raw expression values
                #print("\nRaw expression values for first gene:")
                #first_gene = genes[0]
                #for t in target_times:
                #    expr = self.df[self.df['Gene1_clean'] == first_gene][f'Gene1_Time_{t}'].values
                #   if len(expr) == 0:
                #        expr = self.df[self.df['Gene2_clean'] == first_gene][f'Gene2_Time_{t}'].values
                #    print(f"Time {t}: {expr[0] if len(expr) > 0 else 'Not found'}")
            
            sequences.append(seq_graphs)
            labels.append(label_graphs)
            #print(f"Labels: {labels}")
        
        print(f"\nCreated {len(sequences)} sequences")

        return sequences, labels
    
def process_batch(seq, label):
    # Stack sequence [seq_len, num_nodes, features]
    x = torch.stack([g.x for g in seq])  # [5, 52, 32]
    
    # Reshape for STGCN [batch, channels, time_steps, nodes]
    x = x.permute(2, 0, 1).unsqueeze(0)  # [1, 32, 5, 52]  # Fixed typo
    
    # Process target - keeping temporal dimension the same
    target = torch.stack([g.x for g in label])  # [seq_len, 52, 32]
    target = target.permute(2, 0, 1).unsqueeze(0)  # [1, 32, seq_len, 52]
    
    return x, target

def calculate_correlation(tensor):
    # tensor shape: [batch, channels, time, nodes]
    # Reshape to 2D for correlation
    tensor = tensor.squeeze(0) # remove batch
    tensor = tensor.view(tensor.size(0), -1) # [channels, time*nodes]
    return torch.corrcoef(tensor)

def train_stgcn(dataset, val_ratio=0.2):

    args = Args()
    args.n_vertex = dataset.num_nodes

    print(f"\nModel Configuration:")
    print(f"Number of nodes: {args.n_vertex}")
    print(f"Historical sequence length: {args.n_his}")
    print(f"Block structure: {args.blocks}")

    # sequences with labels
    sequences, labels = dataset.get_temporal_sequences()
    print(f"\nCreated {len(sequences)} sequences")

    # calculate GSO
    edge_index = sequences[0][0].edge_index
    edge_weight = sequences[0][0].edge_attr.squeeze() if sequences[0][0].edge_attr is not None else None

    adj = torch.zeros((args.n_vertex, args.n_vertex)) # symmetric matrix
    adj[edge_index[0], edge_index[1]] = 1 if edge_weight is None else edge_weight # diagonal vs nondiagonal elements for adj matrix
    D = torch.diag(torch.sum(adj, dim=1) ** (-0.5))
    args.gso = torch.eye(args.n_vertex) - D @ adj @ D

    # Split data into train and validation
    n_samples = len(sequences)
    n_train = int(n_samples * (1 - val_ratio))
    train_sequences = sequences[:n_train]
    train_labels = labels[:n_train]
    val_sequences = sequences[n_train:]
    val_labels = labels[n_train:]

    print(f"\nData Split:")
    print(f"Training sequences: {len(train_sequences)}")
    print(f"Validation sequences: {len(val_sequences)}")

    # Initialize STGCN
    model = STGCNChebGraphConv(args, args.blocks, args.n_vertex)
    model = model.float() # convert model to float otherwise I am getting type error

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    num_epochs = 100
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    save_dir = 'plottings_STGCN'
    os.makedirs(save_dir, exist_ok=True)

    train_losses = []
    val_losses = []
    interaction_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for seq,label in zip(train_sequences, train_labels):
            optimizer.zero_grad()

            x, target = process_batch(seq,label)
            output = model(x)

            target = target[:,:,-1:, :]
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        epoch_interaction_loss = 0

        with torch.no_grad():
            for seq,label in zip(val_sequences, val_labels):
                x, target = process_batch(seq,label)
                output = model(x)

                target = target[:,:,-1:, :]
                val_loss = criterion(output, target)

                output_corr = calculate_correlation(output)
                target_corr = calculate_correlation(target)
                int_loss = F.mse_loss(output_corr, target_corr)
                epoch_interaction_loss += int_loss.item()
        
        # Calculate average losses
        avg_train_loss = total_loss / len(train_sequences)
        avg_val_loss = val_loss / len(val_sequences)
        avg_interaction_loss = epoch_interaction_loss / len(val_sequences)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        interaction_losses.append(avg_interaction_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        print(f'Interaction Loss: {avg_interaction_loss:.4f}\n')
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, f'{save_dir}/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Load best model
    checkpoint = torch.load(f'{save_dir}/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Plot training progress
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(interaction_losses, label='Interaction Loss')
    plt.title('Interaction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_progress.png')
    plt.close()
    
    return model, train_losses, val_losses  










def analyze_label_distribution(model, val_sequences, val_labels, dataset):
    """Analyze the distribution of label values"""
    print("\nAnalyzing label value distribution...")
    
    all_labels = []
    for label_seq in val_labels:
        #label_tensor = torch.stack([g.x for g in label_seq]).mean(dim=0)
        label_tensor = torch.stack([g.x for g in label_seq]).squeeze(dim=0)
        all_labels.append(label_tensor.numpy())
    
    all_labels = np.concatenate(all_labels, axis=0)
    
    print(f"Label statistics:")
    print(f"Mean: {np.mean(all_labels):.4f}")
    print(f"Std: {np.std(all_labels):.4f}")
    print(f"Min: {np.min(all_labels):.4f}")
    print(f"Max: {np.max(all_labels):.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.hist(all_labels.flatten(), bins=50)
    plt.title("Distribution of Label Values")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig(f'plottings_STGCN_temporal_graph/label_distribution.png')
    plt.close()


def evaluate_model_performance(model, val_sequences, val_labels, dataset, split_index, save_dir='plottings_STGCN_temporal_graph'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    metrics = {}
   
    time_points = dataset.time_points
    seq_length = dataset.seq_len
    start_idx = split_index + seq_length
    prediction_times = time_points[start_idx:]
    
    print("\nValidation Time Information:")
    print(f"Original time points: {time_points}")
    print(f"Split index: {split_index}")
    print(f"Sequence length: {seq_length}")
    print(f"Start index for predictions: {start_idx}")
    print(f"Prediction times: {prediction_times}")

    with torch.no_grad():
        all_predictions = []
        all_targets = []
        
        for seq, label in zip(val_sequences, val_labels):
            pred = model(seq)
            #target = torch.stack([g.x for g in label]).mean(dim=0)
            target = torch.stack([g.x for g in label]).squeeze(dim=0)
            all_predictions.append(pred.numpy())
            all_targets.append(target.numpy())
        
        predictions = np.stack(all_predictions)
        targets = np.stack(all_targets)
        
        # 1. Overall Time Series Metrics
        metrics['Overall'] = {
            'MSE': np.mean((predictions - targets) ** 2),
            'MAE': np.mean(np.abs(predictions - targets)),
            'RMSE': np.sqrt(np.mean((predictions - targets) ** 2)),
            'Pearson_Correlation': pearsonr(predictions.flatten(), targets.flatten())[0],
            'R2_Score': 1 - np.sum((predictions - targets) ** 2) / np.sum((targets - np.mean(targets)) ** 2)
        }
        
        genes = list(dataset.node_map.keys())
        gene_correlations = []
        
        for i, gene in enumerate(genes):
            gene_pred = predictions[:, i, :]
            #print(f"gene_pred.shape: {gene_pred.shape}")
            gene_target = targets[:, i, :]
            #print(f"gene_target.shape: {gene_target.shape}")
            corr = pearsonr(gene_pred.flatten(), gene_target.flatten())[0]
            gene_correlations.append(corr)
            
            plt.figure(figsize=(10, 6))
            plt.plot(gene_target[0], label='Actual', alpha=0.7)
            plt.plot(gene_pred[0], label='Predicted', alpha=0.7)
            plt.title(f'{gene} Prediction (corr={corr:.3f})')
            plt.legend()
            plt.savefig(os.path.join(save_dir, f'{gene}_prediction.png'))
            plt.close()
        
        metrics['Gene_Performance'] = {
            'Mean_Correlation': np.mean(gene_correlations),
            'Median_Correlation': np.median(gene_correlations),
            'Std_Correlation': np.std(gene_correlations),
            'Best_Genes': [genes[i] for i in np.argsort(gene_correlations)[-5:]],
            'Worst_Genes': [genes[i] for i in np.argsort(gene_correlations)[:5]]
        }
        
        temporal_corrs = []
        for t in range(predictions.shape[1]):
            corr = pearsonr(predictions[:, t, :].flatten(), 
                          targets[:, t, :].flatten())[0]
            temporal_corrs.append(corr)
        
        metrics['Temporal_Stability'] = {
            'Mean_Temporal_Correlation': np.mean(temporal_corrs),
            'Std_Temporal_Correlation': np.std(temporal_corrs),
            'Min_Temporal_Correlation': np.min(temporal_corrs),
            'Max_Temporal_Correlation': np.max(temporal_corrs)
        }
        
        with open(os.path.join(save_dir, 'full_evaluation.txt'), 'w') as f:
            f.write("Model Evaluation Results\n\n")
            for category, category_metrics in metrics.items():
                f.write(f"\n{category}:\n")
                for metric, value in category_metrics.items():
                    if isinstance(value, (float, int)):
                        f.write(f"{metric}: {value:.4f}\n")
                    else:
                        f.write(f"{metric}: {value}\n")
        
        return metrics
    
def evaluate_model_with_direct_values(model, val_sequences, val_labels, dataset, save_dir='plottings_STGCN_temporal_graph'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # Target times for predictions
    target_times = [21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0]
    
    with torch.no_grad():
        all_predictions = []
        all_targets = []
        
        # Get predictions without averaging
        for seq, label in zip(val_sequences, val_labels):
            pred = model(seq)
            target = label[0].x  # Direct access to features
            all_predictions.append(pred.numpy())
            all_targets.append(target.numpy())
            
            # Print shapes for debugging
            print(f"\nPrediction shape: {pred.shape}")
            print(f"Target shape: {target.shape}")
        
        # Stack predictions and targets
        predictions = np.stack(all_predictions)
        targets = np.stack(all_targets)
        
        # Plot for each gene
        genes = list(dataset.node_map.keys())
        for i, gene in enumerate(genes):
            # Get gene's predictions and targets
            # Use first feature column for plotting
            gene_pred = predictions[:, i, 0]  # [num_sequences]
            gene_target = targets[:, i, 0]    # [num_sequences]
            
            plt.figure(figsize=(12, 6))
            
            # Plot with actual time points
            plt.plot(target_times, gene_target, 'bo-', label='Actual', markersize=8)
            plt.plot(target_times, gene_pred, 'ro--', label='Predicted', markersize=8)
            
            plt.title(f'Gene: {gene}')
            plt.xlabel('Time Points')
            plt.ylabel('Expression Value')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Add value labels
            for t, pred, actual in zip(target_times, gene_pred, gene_target):
                plt.annotate(f'{actual:.2f}', (t, actual),
                           textcoords="offset points", xytext=(0,10),
                           ha='center', color='blue')
                plt.annotate(f'{pred:.2f}', (t, pred),
                           textcoords="offset points", xytext=(0,-15),
                           ha='center', color='red')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{gene}_prediction.png'), dpi=300)
            plt.close()
            
            if i == 0:
                print(f"\nValues for gene {gene}:")
                print("Time points:", target_times)
                print("Predicted:", gene_pred)
                print("Actual:", gene_target)
    
class Args:
    def __init__(self):
        self.Kt = 3  # temporal kernel size
        self.Ks = 3  # spatial kernel size
        self.n_his = 5  # historical sequence length
        self.n_pred = 1
        # Modified block structure with fewer ST blocks
        self.blocks = [
            [32, 32, 32],    # Input block
            [32, 32, 32],    # Single ST block (since temporal dim reduces quickly)
            [32, 32, 1]      # Output block
        ]
        self.act_func = 'glu'
        self.graph_conv_type = 'cheb_graph_conv'
        self.enable_bias = True
        self.droprate = 0.3

if __name__ == "__main__":
    dataset = TemporalGraphDataset(
        csv_file='mapped/enhanced_interactions.csv',
        embedding_dim=32,
        seq_len=5,
        pred_len=1
    )
    
    model, train_losses, val_losses = train_stgcn(dataset, val_ratio=0.2)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('STGCN Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('stgcn_training_progress.png')
    plt.close()