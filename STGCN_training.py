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
from STGCN.model.models import STGCNChebGraphConv
import argparse
from scipy.spatial.distance import cdist
from create_graph_and_embeddings_STGCN import *
    
def process_batch(seq, label):
    """Process batch data for training."""
    print("\n=== Input Sequence Statistics ===")
    x = torch.stack([g.x for g in seq])
    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"Input mean: {x.mean():.4f}")
    x = x.permute(2, 0, 1).unsqueeze(0)  # [1, channels, time_steps, nodes]
    
    print("\n=== Target Statistics ===")
    target = torch.stack([g.x for g in label])
    print(f"Target shape: {target.shape}")
    print(f"Target range: [{target.min():.4f}, {target.max():.4f}]")
    print(f"Target mean: {target.mean():.4f}")

    target = target.permute(2, 0, 1).unsqueeze(0)  # [1, channels, time_steps, nodes]
    
    # Take only the last time step of target
    target = target[:, :, -1:, :]  # [1, channels, 1, nodes]
    
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
    model = STGCNChebGraphConv(args, args.blocks, args.n_vertex)
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
            target = target[:, :, -1:, :]  # Keep only the last timestep
            output = output[:, :, -1:, :]
            batch_stats.append({
                'target_range': [target.min().item(), target.max().item()],
                'output_range': [output.min().item(), output.max().item()],
                'target_mean': target.mean().item(),
                'output_mean': output.mean().item()
            })

            all_targets.append(target.detach().cpu().numpy())
            all_outputs.append(output.detach().cpu().numpy())
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 5 == 0:
            targets = np.concatenate(all_targets)
            outputs = np.concatenate(all_outputs)
            print(f"\nEpoch {epoch} Detailed Statistics:")
            print(f"Target range: [{targets.min():.4f}, {targets.max():.4f}]")
            print(f"Target mean: {targets.mean():.4f}")
            print(f"Output range: [{outputs.min():.4f}, {outputs.max():.4f}]")
            print(f"Output mean: {outputs.mean():.4f}")
            print(f"Loss: {total_loss/len(train_sequences):.4f}")

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
                target = target[:, :, -1:, :]  
                output = output[:, :, -1:, :]
                #print(f"Shape of output in validation: {output.shape}") # --> [1, 32, 5, 52]
                #print(f"Shape of target in validation: {target.shape}") # --> [32, 1, 52]
                #target = target[:,:,-1:, :]
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

def analyze_interactions(model, val_sequences, val_labels, dataset):
    """Analyzes the preservation of gene-gene interactions."""
    model.eval()
    all_predicted = []
    all_true = []
    
    with torch.no_grad():
        for seq, label in zip(val_sequences, val_labels):
            x, target = process_batch(seq, label)  # Shape: [1, 32, seq_len, 52]
            output = model(x)  # Shape: [1, 32, seq_len, 52]
            
            # Extract predicted interactions (correlation between genes)
            predicted_corr = calculate_correlation(output)  # [channels, channels]
            true_corr = calculate_correlation(target)  # [channels, channels]
            
            # Collect results for interaction loss
            all_predicted.append(predicted_corr.cpu().numpy())
            all_true.append(true_corr.cpu().numpy())
    
    predicted_interactions = np.concatenate(all_predicted, axis=0)
    true_interactions = np.concatenate(all_true, axis=0)
    
    interaction_loss = np.mean((predicted_interactions - true_interactions) ** 2)
    
    print(f"Interaction Preservation (MSE between predicted and true interaction matrix): {interaction_loss:.4f}")
    
    return interaction_loss

def analyze_gene(model, val_sequences, val_labels, dataset):
    """Analyzes the gene prediction performance over time for validation sequences."""
    model.eval()
    all_predicted = []
    all_true = []
    
    with torch.no_grad():
        for seq, label in zip(val_sequences, val_labels):
            x, target = process_batch(seq, label)  # Shape: [1, 32, seq_len, 52]
            output = model(x)  # Shape: [1, 32, seq_len, 52]
            
            # remove the batch dimension, keeping seq_len and gene dimensions
            all_predicted.append(output.squeeze(0).cpu().numpy())  # Shape: [seq_len, genes]
            all_true.append(target.squeeze(0).cpu().numpy())  # Shape: [seq_len, genes]
    
    all_predicted = np.concatenate(all_predicted, axis=0)  # Shape: [total_time_steps, genes]
    all_true = np.concatenate(all_true, axis=0)  # Shape: [total_time_steps, genes]

    if all_predicted.shape[1] != all_true.shape[1]:
        # Align shapes by duplicating the true labels across all predicted genes
        if all_true.shape[1] == 1 and all_predicted.shape[1] > 1:
            all_true = np.repeat(all_true, all_predicted.shape[1], axis=1) 
        else:
            raise ValueError(f"Predicted and true arrays have different gene dimensions: "
                             f"{all_predicted.shape[1]} vs {all_true.shape[1]}")
    
    gene_corrs = []
    num_genes = all_true.shape[1]  
    
    for gene_idx in range(num_genes):  # Iterate over genes
        # Flatten the arrays to 1D
        pred_gene = all_predicted[:, gene_idx].flatten()  # Flatten to 1D
        true_gene = all_true[:, gene_idx].flatten()  # Flatten to 1D
        
        corr, _ = pearsonr(pred_gene, true_gene)
        gene_corrs.append(corr)
    
    mean_corr = np.mean(gene_corrs)
    
    print(f"Mean Pearson Correlation between predicted and true gene expressions: {mean_corr:.4f}")
    
    return mean_corr

def evaluate_model_performance(model, val_sequences, val_labels, dataset, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    metrics = {
        'Overall': {},
        'Gene_Performance': {},
        'Temporal_Stability': {}
    }
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for seq, label in zip(val_sequences, val_labels):
            x, target = process_batch_evaluation(seq, label)
            output = model(x)
            
            # Take only the last time step of predictions to match target
            output = output[:, :, -1:, :]  # Shape: [1, channels, 1, nodes]
            
            # Reshape for evaluation
            pred = output.squeeze().cpu().numpy()  # Shape: [channels, nodes]
            true = target.squeeze().cpu().numpy()  # Shape: [channels, nodes]
            
            all_predictions.append(pred)
            all_targets.append(true)
    
    # Stack all predictions and targets
    predictions = np.stack(all_predictions, axis=0)  # [samples, channels, nodes]
    targets = np.stack(all_targets, axis=0)  # [samples, channels, nodes]
    
    # Calculate overall metrics
    metrics['Overall'] = calculate_overall_metrics(predictions, targets)
    
    # Calculate gene-wise metrics
    metrics['Gene_Performance'] = calculate_gene_metrics(
        predictions, targets, dataset.node_map, save_dir
    )
    
    # Calculate temporal stability metrics
    metrics['Temporal_Stability'] = calculate_temporal_metrics(
        predictions, targets
    )
    
    # Create visualizations
    create_evaluation_plots(predictions, targets, dataset, save_dir)
    gene_summary = create_detailed_gene_plots(predictions, targets, dataset, 
                                            save_dir=f'{save_dir}/gene_plots')
    
    return metrics

def calculate_overall_metrics(predictions, targets):
    """Calculate overall model performance metrics."""
    metrics = {}
    
    # Reshape arrays for overall metrics
    pred_flat = predictions.reshape(-1)
    target_flat = targets.reshape(-1)
    
    # Basic regression metrics
    metrics['MSE'] = mean_squared_error(target_flat, pred_flat)
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    metrics['MAE'] = mean_absolute_error(target_flat, pred_flat)
    metrics['R2_Score'] = r2_score(target_flat, pred_flat)
    
    # Overall correlation
    metrics['Pearson_Correlation'], _ = pearsonr(target_flat, pred_flat)
    
    return metrics

def calculate_gene_metrics(predictions, targets, node_map, save_dir):
    """Calculate gene-wise performance metrics."""
    metrics = {}
    gene_correlations = []
    gene_rmse = []
    save_dir='plottings_STGCN'
    
    num_genes = predictions.shape[-1]
    genes = list(node_map.keys())
    
    for gene_idx in range(num_genes):
        pred_gene = predictions[..., gene_idx].flatten()
        true_gene = targets[..., gene_idx].flatten()
        
        # Handle NaN values
        mask = ~(np.isnan(pred_gene) | np.isnan(true_gene))
        pred_gene = pred_gene[mask]
        true_gene = true_gene[mask]
        
        if len(pred_gene) > 0:
            corr, _ = pearsonr(pred_gene, true_gene)
            rmse = np.sqrt(mean_squared_error(true_gene, pred_gene))
        else:
            corr = np.nan
            rmse = np.nan
        
        gene_correlations.append(corr)
        gene_rmse.append(rmse)
    
    # Filter out NaN values for sorting
    valid_performances = [(gene, corr) for gene, corr in zip(genes, gene_correlations) 
                         if not np.isnan(corr)]
    valid_performances.sort(key=lambda x: x[1], reverse=True)
    
    metrics['Mean_Correlation'] = np.nanmean(gene_correlations)
    metrics['Mean_RMSE'] = np.nanmean(gene_rmse)
    metrics['Best_Genes'] = [gene for gene, _ in valid_performances[:5]]
    metrics['Worst_Genes'] = [gene for gene, _ in valid_performances[-5:]]
    
    # Create gene performance visualization
    valid_correlations = [c for c in gene_correlations if not np.isnan(c)]
    valid_rmse = [r for r in gene_rmse if not np.isnan(r)]
    
    plt.figure(figsize=(12, 8))
    plt.scatter(valid_correlations, valid_rmse, alpha=0.5)
    plt.xlabel('Correlation')
    plt.ylabel('RMSE')
    plt.title('Gene Performance Distribution')
    plt.show()
    plt.savefig(f'{save_dir}/gene_performance_distribution.png')
    plt.close()
    
    return metrics

def calculate_temporal_metrics(true_values, predicted_values):

    metrics = {}
    
    # 1. Dynamic Time Warping (DTW) distance
    dtw_distances = []
    for i in range(true_values.shape[1]):
        true_seq = true_values[:, i].reshape(-1, 1)
        pred_seq = predicted_values[:, i].reshape(-1, 1)
        D = cdist(true_seq, pred_seq)
        dtw_distances.append(D.sum())
    metrics['dtw_mean'] = np.mean(dtw_distances)
    
    # 2. Temporal Correlation (considers lag relationships)
    def temporal_corr(y_true, y_pred, max_lag=3):
        correlations = []
        for lag in range(max_lag + 1):
            if lag == 0:
                corr = np.corrcoef(y_true, y_pred)[0, 1]
                #print(f"I'm inside lag=0")
            else:
                corr = np.corrcoef(y_true[lag:], y_pred[:-lag])[0, 1]
            correlations.append(corr)
        return np.max(correlations)  # Return max correlation across lags
    
    temp_corrs = []
    for i in range(true_values.shape[1]):
        temp_corr = temporal_corr(true_values[:, i], predicted_values[:, i])
        temp_corrs.append(temp_corr)
    metrics['temporal_correlation'] = np.mean(temp_corrs)
    
    # 3. Trend Consistency (direction of changes)
    def trend_accuracy(y_true, y_pred):
        true_diff = np.diff(y_true)
        pred_diff = np.diff(y_pred)
        true_direction = np.sign(true_diff)
        pred_direction = np.sign(pred_diff)
        return np.mean(true_direction == pred_direction)
    
    trend_accs = []
    for i in range(true_values.shape[1]):
        trend_acc = trend_accuracy(true_values[:, i], predicted_values[:, i])
        trend_accs.append(trend_acc)
    metrics['trend_accuracy'] = np.mean(trend_accs)
    
    # 4. RMSE over time
    time_rmse = []
    for t in range(true_values.shape[0]):
        rmse = np.sqrt(np.mean((true_values[t] - predicted_values[t])**2))
        time_rmse.append(rmse)
    metrics['rmse_over_time'] = np.mean(time_rmse)
    metrics['rmse_std'] = np.std(time_rmse)
    
    # 5. Temporal Pattern Similarity
    def pattern_similarity(y_true, y_pred):
        # Normalize sequences
        y_true_norm = y_true
        y_pred_norm = y_pred
        # Calculate similarity
        return np.mean(np.abs(y_true_norm - y_pred_norm))
    
    pattern_sims = []
    for i in range(true_values.shape[1]):
        sim = pattern_similarity(true_values[:, i], predicted_values[:, i])
        pattern_sims.append(sim)
    metrics['pattern_similarity'] = np.mean(pattern_sims)
    
    return metrics

def create_evaluation_plots(predictions, targets, dataset, save_dir):
    """Create comprehensive evaluation visualizations."""
    save_dir='plottings_STGCN'

    # 1. Overall prediction scatter plot
    plt.figure(figsize=(10, 10))
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    mask = ~(np.isnan(pred_flat) | np.isnan(target_flat))
    plt.scatter(target_flat[mask], pred_flat[mask], alpha=0.1)
    plt.plot([min(target_flat[mask]), max(target_flat[mask])], 
             [min(target_flat[mask]), max(target_flat[mask])], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Overall Prediction Performance')
    plt.show()
    plt.savefig(f'{save_dir}/overall_scatter.png')
    plt.close()
    
    # 2. Temporal predictions for top genes
    top_genes = list(dataset.node_map.keys())[:5]
    plt.figure(figsize=(15, 10))
    
    for i, gene in enumerate(top_genes):
        gene_idx = dataset.node_map[gene]
        plt.subplot(3, 2, i+1)
        plt.plot(targets[:, 0, gene_idx], label='True', marker='o')
        plt.plot(predictions[:, 0, gene_idx], label='Predicted', marker='s')
        plt.title(f'Gene: {gene}')
        plt.xlabel('Time Step')
        plt.ylabel('Expression')
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{save_dir}/temporal_predictions.png')
    plt.close()
    
    # 3. Correlation heatmap
    pred_corr = np.corrcoef(predictions.mean(axis=0).T)
    true_corr = np.corrcoef(targets.mean(axis=0).T)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    sns.heatmap(true_corr, ax=ax1, cmap='coolwarm')
    ax1.set_title('True Gene Correlations')
    
    sns.heatmap(pred_corr, ax=ax2, cmap='coolwarm')
    ax2.set_title('Predicted Gene Correlations')
    
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{save_dir}/correlation_heatmaps.png')
    plt.close()

def process_batch_evaluation(seq, label):
    """Process batch data for evaluation."""
    x = torch.stack([g.x for g in seq])
    x = x.permute(2, 0, 1).unsqueeze(0)
    
    target = torch.stack([g.x for g in label])
    target = target.permute(2, 0, 1).unsqueeze(0)
    
    return x, target

def create_detailed_gene_plots(predictions, targets, dataset, save_dir):
    save_dir='plottings_STGCN'
    os.makedirs(save_dir, exist_ok=True)
    genes = list(dataset.node_map.keys())
    num_genes = len(genes)

    genes_per_page = 5
    num_pages = (num_genes + genes_per_page - 1) // genes_per_page

    for page in range(num_pages):
        start_idx = page * genes_per_page
        end_idx = min((page + 1) * genes_per_page, num_genes)
        current_genes = genes[start_idx:end_idx]

        fig, axes = plt.subplots(5, 4, figsize=(10, 20))
        fig.suptitle(f'Gene Predictions vs Actuals (Page {page+1}/{num_pages})', fontsize=16)
        axes = axes.flatten()

        for i,gene in enumerate(current_genes):
            gene_idx = dataset.node_map[gene]
            pred = predictions[:,0,gene_idx]
            true = targets[:,0, gene_idx]

            if not (np.isnan(pred).any() or np.isnan(true).any()):
                corr, p_value = pearsonr(pred, true)
                corr_text = f'Correlation: {corr:.3f}\np-value: {p_value:.3e}'
            else:
                corr_text = 'Correlation: N/A'

            # Create subplot
            ax = axes[i]
            ax.plot(true, label='Actual', marker='o')
            ax.plot(pred, label='Predicted', marker='s')
            ax.set_title(f'Gene: {gene}\n{corr_text}')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Expression Level')
            ax.legend()
            ax.grid(True)

            rmse = np.sqrt(np.mean((pred - true) ** 2))
            ax.text(0.05, 0.95, f'RMSE: {rmse:.3f}', 
                   transform=ax.transAxes, 
                   verticalalignment='top')
        
        for j in range(len(current_genes), len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/gene_predictions_page_{page+1}.png', 
                   bbox_inches='tight', dpi=300)
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
        seq_len=5,
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

    print("\nModel Performance Summary:")
    print(f"Overall Metrics:")
    print(f"Pearson Correlation: {metrics['Overall']['Pearson_Correlation']:.4f}")
    print(f"RMSE: {metrics['Overall']['RMSE']:.4f}")
    print(f"R² Score: {metrics['Overall']['R2_Score']:.4f}")
    
    print(f"\nGene-wise Performance:")
    print(f"Mean Gene Correlation: {metrics['Gene_Performance']['Mean_Correlation']:.4f}")
    print(f"Best Performing Genes: {', '.join(metrics['Gene_Performance']['Best_Genes'])}")
    
    print(f"\nTemporal Performance:")
    print(f"DTW Distance: {metrics['Temporal_Stability']['dtw_mean']:.4f}")
    print(f"Temporal Correlation: {metrics['Temporal_Stability']['temporal_correlation']:.4f}")
    print(f"Pattern Similarity: {metrics['Temporal_Stability']['pattern_similarity']:.4f}")