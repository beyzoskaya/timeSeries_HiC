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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
sys.path.append('./STGCN')
from STGCN.model.models import STGCNChebGraphConv, STGCNChebGraphConvProjected
import argparse
from scipy.spatial.distance import cdist
from create_graph_and_embeddings_STGCN import *
from STGCN_losses import temporal_pattern_loss, change_magnitude_loss
from evaluation import *
    
def process_batch(seq, label):
    """Process batch data for training."""
    # Input: Use full embeddings
    x = torch.stack([g.x for g in seq])  # [seq_len, num_nodes, features]
    x = x.permute(2, 0, 1).unsqueeze(0)  # [1, features, seq_len, nodes]
    
    # Target: Use only expression values
    target = torch.stack([g.x[:, -1] for g in label])  # [1, nodes] (expression values)
    target = target.unsqueeze(1).unsqueeze(0)  # [1, 1, 1, nodes]
    
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

    #print(f"\nModel Configuration:")
    #print(f"Number of nodes: {args.n_vertex}")
    #print(f"Historical sequence length: {args.n_his}")
    #print(f"Block structure: {args.blocks}")

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
    
    """
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
    """
    # Random split
    n_samples = len(sequences)
    n_train = int(n_samples * (1 - val_ratio))
    indices = torch.randperm(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    train_sequences = [sequences[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_sequences = [sequences[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    print("\n=== Training Data Statistics ===")
    print(f"Number of training sequences: {len(train_sequences)}")
    print(f"Number of validation sequences: {len(val_sequences)}")
    
    # Initialize STGCN
    model = STGCNChebGraphConvProjected(args, args.blocks, args.n_vertex)
    model = model.float() # convert model to float otherwise I am getting type error

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
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
        batch_stats = []
        all_targets = []
        all_outputs = []

        for seq,label in zip(train_sequences, train_labels):
            optimizer.zero_grad()
            x,target = process_batch(seq, label)
            #x, _ = process_batch(seq, label)
            #print(f"Shape of x inside training: {x.shape}") # --> [1, 32, 5, 52]
            #print(f"Shape of target inside training: {target.shape}") # --> [1, 32, 1, 52]
            output = model(x)
            #print(f"Shape of output: {output.shape}") # --> [1, 32, 5, 52]
            #_, target = process_batch(seq, label)
            #target = target[:,:,-1:, :]
            #print(f"Shape of target: {target.shape}") # --> [32, 1, 52]

            # Don't take the last point for temporal loss !!!!!!
            #target = target[:, :, -1:, :]  # Keep only the last timestep
            output = output[:, :, -1:, :]

            batch_stats.append({
                'target_range': [target.min().item(), target.max().item()],
                'output_range': [output.min().item(), output.max().item()],
                'target_mean': target.mean().item(),
                'output_mean': output.mean().item()
            })

            all_targets.append(target.detach().cpu().numpy())
            all_outputs.append(output.detach().cpu().numpy())
            #loss = temporal_pattern_loss(output[:, :, -1:, :], target, x)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        #if epoch % 5 == 0:
        #    targets = np.concatenate(all_targets)
        #    outputs = np.concatenate(all_outputs)
        #    print(f"\nEpoch {epoch} Detailed Statistics:")
        #    print(f"Target range: [{targets.min():.4f}, {targets.max():.4f}]")
        #    print(f"Target mean: {targets.mean():.4f}")
        #    print(f"Output range: [{outputs.min():.4f}, {outputs.max():.4f}]")
        #    print(f"Output mean: {outputs.mean():.4f}")
        #    print(f"Loss: {total_loss/len(train_sequences):.4f}")

        model.eval()
        val_loss = 0
        val_loss_total = 0
        epoch_interaction_loss = 0
        with torch.no_grad():
            for seq,label in zip(val_sequences, val_labels):
                #x, _ = process_batch(seq, label)
                x,target = process_batch(seq, label)
                
                output = model(x)
                #_, target = process_batch(seq, label)

                # Don't take the last point for temporal loss!!!
                #target = target[:, :, -1:, :]  
                output = output[:, :, -1:, :]

                #print(f"Shape of output in validation: {output.shape}") # --> [1, 32, 5, 52]
                #print(f"Shape of target in validation: {target.shape}") # --> [32, 1, 52]
                #target = target[:,:,-1:, :]
                #val_loss = temporal_pattern_loss(output[:, :, -1:, :], target, x)
                val_loss = criterion(output, target)
                val_loss_total += val_loss.item()

                output_corr = calculate_correlation(output)
                #print(f"Shape of output corr: {output_corr.shape}") # [32, 32]
                target_corr = calculate_correlation(target)
                #print(f"Shape of target corr: {target_corr.shape}") # [32, 32]
                int_loss = F.mse_loss(output_corr, target_corr)
                epoch_interaction_loss += int_loss.item()
        
        # Calculate average losses
        avg_train_loss = total_loss / len(train_sequences)
        avg_val_loss = val_loss_total / len(val_sequences)
        avg_interaction_loss = epoch_interaction_loss / len(val_sequences)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        interaction_losses.append(avg_interaction_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        #print(f'Interaction Loss: {avg_interaction_loss:.4f}\n')
        
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
    checkpoint = torch.load(f'{save_dir}/best_model.pth', weights_only=True)
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
    plt.show()
    plt.savefig(f'{save_dir}/training_progress.png')
    plt.close()
    
    return model, val_sequences, val_labels, train_losses, val_losses  

def evaluate_model_performance(model, val_sequences, val_labels, dataset, save_dir='plottings_STGCN'):

    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for seq, label in zip(val_sequences, val_labels):
            # Get input and target
            x, target = process_batch(seq, label)
            output = model(x)
            
            # Extract only the expression predictions
            output = output[:, :, -1:, :].squeeze().cpu().numpy()  # [nodes] expression values
            target = target.squeeze().cpu().numpy()  # [nodes] expression values
            
            all_predictions.append(output)
            all_targets.append(target)
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions)  # [time_points, nodes]
    targets = np.array(all_targets)      # [time_points, nodes]
    
    # Calculate metrics
    overall_metrics = calculate_overall_metrics(predictions, targets)
    gene_metrics = calculate_gene_metrics(predictions, targets, dataset)
    temporal_metrics = calculate_temporal_metrics(predictions, targets, dataset)
    
    # Create visualizations
    create_evaluation_plots(predictions, targets, dataset, save_dir)
    
    metrics = {
        'Overall': overall_metrics,
        'Gene': gene_metrics,
        'Temporal': temporal_metrics
    }
    
    return metrics

def calculate_overall_metrics(predictions, targets):
    """Calculate overall expression prediction metrics."""
    metrics = {}
    
    # Flatten for overall metrics
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    # Calculate basic metrics
    metrics['MSE'] = mean_squared_error(target_flat, pred_flat)
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    metrics['MAE'] = mean_absolute_error(target_flat, pred_flat)
    metrics['R2_Score'] = r2_score(target_flat, pred_flat)
    metrics['Pearson_Correlation'], _ = pearsonr(target_flat, pred_flat)
    
    return metrics

def calculate_gene_metrics(predictions, targets, dataset):
    """Calculate gene-specific metrics."""
    metrics = {}
    genes = list(dataset.node_map.keys())
    
    # Per-gene correlations
    gene_correlations = []
    gene_rmse = []
    
    for gene_idx, gene in enumerate(genes):
        pred_gene = predictions[:, gene_idx]  # All timepoints for this gene
        true_gene = targets[:, gene_idx]
        
        corr, _ = pearsonr(pred_gene, true_gene)
        rmse = np.sqrt(mean_squared_error(true_gene, pred_gene))
        
        gene_correlations.append((gene, corr))
        gene_rmse.append(rmse)
    
    # Sort genes by correlation
    gene_correlations.sort(key=lambda x: x[1], reverse=True)
    
    metrics['Mean_Correlation'] = np.mean([corr for _, corr in gene_correlations])
    metrics['Best_Genes'] = [gene for gene, _ in gene_correlations[:5]]
    metrics['Gene_RMSE'] = {gene: rmse for gene, rmse in zip(genes, gene_rmse)}
    
    return metrics

def calculate_temporal_metrics(predictions, targets, dataset):
    """Calculate temporal prediction metrics."""
    metrics = {}
    
    # Calculate changes between consecutive timepoints
    true_changes = np.diff(targets, axis=0)  # [time-1, nodes]
    pred_changes = np.diff(predictions, axis=0)
    
    # Direction accuracy (whether changes are in the same direction)
    direction_match = np.sign(true_changes) == np.sign(pred_changes)
    metrics['Direction_Accuracy'] = np.mean(direction_match)
    
    # Magnitude of changes
    metrics['Mean_True_Change'] = np.mean(np.abs(true_changes))
    metrics['Mean_Pred_Change'] = np.mean(np.abs(pred_changes))
    
    # Temporal correlation per gene
    genes = list(dataset.node_map.keys())
    temporal_corrs = []
    
    for gene_idx, gene in enumerate(genes):
        true_seq = targets[:, gene_idx]
        pred_seq = predictions[:, gene_idx]
        corr, _ = pearsonr(true_seq, pred_seq)
        temporal_corrs.append(corr)
    
    metrics['Mean_Temporal_Correlation'] = np.mean(temporal_corrs)
    
    return metrics

def create_evaluation_plots(predictions, targets, dataset, save_dir):
    """Create comprehensive evaluation plots."""
    # 1. Overall prediction scatter
    plt.figure(figsize=(10, 8))
    plt.scatter(targets.flatten(), predictions.flatten(), alpha=0.1)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
    plt.xlabel('True Expression')
    plt.ylabel('Predicted Expression')
    plt.title('Expression Prediction Performance')
    plt.savefig(f'{save_dir}/overall_scatter.png')
    plt.close()
    
    # 2. Change distribution plot
    true_changes = np.diff(targets, axis=0).flatten()
    pred_changes = np.diff(predictions, axis=0).flatten()
    
    plt.figure(figsize=(12, 6))
    plt.hist(true_changes, bins=50, alpha=0.5, label='Actual Changes')
    plt.hist(pred_changes, bins=50, alpha=0.5, label='Predicted Changes')
    plt.xlabel('Expression Change')
    plt.ylabel('Frequency')
    plt.title('Distribution of Expression Changes')
    plt.legend()
    plt.savefig(f'{save_dir}/change_distribution.png')
    plt.close()
    
    # 3. Gene temporal patterns
    genes = list(dataset.node_map.keys())
    plt.figure(figsize=(15, 10))
    
    for i in range(min(6, len(genes))):
        plt.subplot(2, 3, i+1)
        gene_idx = dataset.node_map[genes[i]]
        
        plt.plot(targets[:, gene_idx], label='Actual', marker='o')
        plt.plot(predictions[:, gene_idx], label='Predicted', marker='s')
        plt.title(f'Gene: {genes[i]}')
        plt.xlabel('Time Step')
        plt.ylabel('Expression')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/temporal_patterns.png')
    plt.close()
    
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
        seq_len=3,
        pred_len=1
    )
    
    model, val_sequences,val_labels, train_losses, val_losses = train_stgcn(dataset, val_ratio=0.2)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('STGCN Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('plottings_STGCN/stgcn_training_progress.png')
    plt.close()

    # After training
    metrics = evaluate_model_performance(
    model, 
    val_sequences, 
    val_labels, 
    dataset,
    save_dir='plottings_STGCN'
    )


    metrics = evaluate_model_performance(model, val_sequences, val_labels, dataset)

    print("\nModel Performance Summary:")
    print("\nOverall Metrics:")
    for metric, value in metrics['Overall'].items():
        print(f"{metric}: {value:.4f}")

    print("\nGene Performance:")
    print(f"Mean Gene Correlation: {metrics['Gene']['Mean_Correlation']:.4f}")
    print(f"Best Performing Genes: {', '.join(metrics['Gene']['Best_Genes'])}")

    print("\nTemporal Performance:")
    print(f"Direction Accuracy: {metrics['Temporal']['Direction_Accuracy']:.4f}")
    print(f"Mean Temporal Correlation: {metrics['Temporal']['Mean_Temporal_Correlation']:.4f}")
    print(f"Average True Change: {metrics['Temporal']['Mean_True_Change']:.4f}")
    print(f"Average Predicted Change: {metrics['Temporal']['Mean_Pred_Change']:.4f}")