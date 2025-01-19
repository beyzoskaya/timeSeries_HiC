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
from STGCN.model.models import STGCNChebGraphConv, STGCNChebGraphConvProjected, STGCNGraphConv, STGCNGraphConvProjected, EnhancedSTGCNChebGraphConvProjected
import argparse
from scipy.spatial.distance import cdist
from create_graph_and_embeddings_STGCN import *
from STGCN_losses import temporal_loss_for_projected_model, temporal_loss_for_projected_model_rolling_window
from evaluation import *
from sklearn.model_selection import TimeSeriesSplit
    
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

def rolling_window_split(sequences, labels, train_size, val_size, step=1):
    for i in range(0, len(sequences) - train_size - val_size + 1, step):
        train_seq = sequences[i:i + train_size]
        train_lbl = labels[i:i + train_size]
        val_seq = sequences[i + train_size:i + train_size + val_size]
        val_lbl = labels[i + train_size:i + train_size + val_size]
        yield train_seq, train_lbl, val_seq, val_lbl

def train_stgcn(dataset, val_ratio=0.2, step=1):
    # ADDED FOR GRID SEARCH ARGS
    args = Args()
    args.n_vertex = dataset.num_nodes

    all_train_losses = []
    all_val_losses = []

    # sequences with labels
    sequences, labels = dataset.get_temporal_sequences()
    print(f"\nCreated {len(sequences)} sequences")

    # calculate GSO
    edge_index = sequences[0][0].edge_index
    edge_weight = sequences[0][0].edge_attr.squeeze() if sequences[0][0].edge_attr is not None else None

    adj = torch.zeros((args.n_vertex, args.n_vertex))  # symmetric matrix
    adj[edge_index[0], edge_index[1]] = 1 if edge_weight is None else edge_weight  # diagonal vs nondiagonal elements for adj matrix
    D = torch.diag(torch.sum(adj, dim=1) ** (-0.5))
    args.gso = torch.eye(args.n_vertex) - D @ adj @ D

    # Define train and validation sizes
    train_size = int(len(sequences) * (1 - val_ratio))  # e.g., 32
    val_size = len(sequences) - train_size  # e.g., 8

    # Use rolling window split
    splits = list(rolling_window_split(sequences, labels, train_size, val_size, step=step))
    if not splits:
        raise ValueError("Not enough sequences to create even one split. Reduce train_size or val_size.")

    # Train and validate for each split
    for split_idx, (train_sequences, train_labels, val_sequences, val_labels) in enumerate(splits):
        print(f"\n=== Split {split_idx + 1} ===")
        print(f"Training sequences: {len(train_sequences)}")
        print(f"Validation sequences: {len(val_sequences)}")

        # Initialize STGCN
        model = STGCNChebGraphConvProjected(args, args.blocks, args.n_vertex)
        model = model.float()  # convert model to float otherwise I am getting type error

        optimizer = optim.Adam(model.parameters(), lr=0.0009)

        num_epochs = 100
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        save_dir = f'plottings_STGCN_split_{split_idx}'
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

            for seq, label in zip(train_sequences, train_labels):
                optimizer.zero_grad()
                x, target = process_batch(seq, label)
                output = model(x)

                batch_stats.append({
                    'target_range': [target.min().item(), target.max().item()],
                    'output_range': [output.min().item(), output.max().item()],
                    'target_mean': target.mean().item(),
                    'output_mean': output.mean().item()
                })

                all_targets.append(target.detach().cpu().numpy())
                all_outputs.append(output.detach().cpu().numpy())
                loss = temporal_loss_for_projected_model_rolling_window(output[:, :, -1:, :], target, x)

                if torch.isnan(loss):
                    print("NaN loss detected!")
                    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
                    print(f"Target range: [{target.min().item():.4f}, {target.max().item():.4f}]")
                    continue

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            model.eval()
            val_loss = 0
            val_loss_total = 0
            epoch_interaction_loss = 0
            with torch.no_grad():
                for seq, label in zip(val_sequences, val_labels):
                    x, target = process_batch(seq, label)
                    output = model(x)
                    val_loss = temporal_loss_for_projected_model_rolling_window(output[:, :, -1:, :], target, x)
                    val_loss_total += val_loss.item()

                    output_corr = calculate_correlation(output)
                    target_corr = calculate_correlation(target)
                    int_loss = F.mse_loss(output_corr, target_corr)
                    epoch_interaction_loss += int_loss.item()

            # Calculate average losses
            avg_train_loss = total_loss / len(train_sequences)
            avg_val_loss = val_loss_total / len(val_sequences)
            avg_interaction_loss = epoch_interaction_loss / len(val_sequences)

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            interaction_losses.append(avg_interaction_loss)

            print(f'Epoch {epoch + 1}/{num_epochs}')
            print(f'Training Loss: {avg_train_loss:.4f}')
            print(f'Validation Loss: {avg_val_loss:.4f}')

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

        # Store results for this split
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

        # Load best model for this split
        checkpoint = torch.load(f'{save_dir}/best_model.pth', weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Plot training progress for this split
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train')
        plt.plot(val_losses, label='Validation')
        plt.title(f'Training and Validation Loss (Split {split_idx + 1})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

    return model, val_sequences, val_labels, all_train_losses, all_val_losses

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
    temporal_metrics = calculate_temporal_metrics_detailly(predictions, targets, dataset)
    
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

def calculate_temporal_metrics_detailly(predictions, targets, dataset):
    """Calculate temporal prediction metrics with more appropriate temporal measures."""
    metrics = {}
    

    """
    What if we shift one sequence by sequence length amount?
    """
    # 1. Time-lagged Cross Correlation
    def time_lagged_correlation(true_seq, pred_seq, max_lag=3):
        correlations = []
        for lag in range(max_lag + 1):
            if lag == 0:
                corr = np.corrcoef(true_seq, pred_seq)[0, 1]
            else:
                corr = np.corrcoef(true_seq[lag:], pred_seq[:-lag])[0, 1]
            correlations.append(corr)
        return np.max(correlations)  # Return max correlation across lags
    
    """
    measures the similarity between two temporal sequences by finding the optimal alignment between them

    True sequence:  [1, 2, 3, 2, 1]
    Pred sequence: [1, 1, 2, 3, 1]

    DTW process:
    1. Creates a matrix of distances
    2. For each point, calculates:
    - Direct cost (difference between values)
    - Adds minimum cost from previous steps
    3. Finds optimal path through matrix that minimizes total distance

    Visual example:
    True:  1 -> 2 -> 3 -> 2 -> 1
            \   |  /  |    /
    Pred:   1 -> 1 -> 2 -> 3 -> 1
    """
    # 2. Dynamic Time Warping Distance
    def dtw_distance(true_seq, pred_seq):
        n, m = len(true_seq), len(pred_seq)
        dtw_matrix = np.zeros((n+1, m+1))
        dtw_matrix[1:, 0] = np.inf
        dtw_matrix[0, 1:] = np.inf
        
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = abs(true_seq[i-1] - pred_seq[j-1])
                dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],    # insertion
                                            dtw_matrix[i, j-1],    # deletion
                                            dtw_matrix[i-1, j-1])  # match
        return dtw_matrix[n, m]
    
    # Calculate temporal metrics for each gene
    genes = list(dataset.node_map.keys())
    temporal_metrics = []
    dtw_distances = []
    direction_accuracies = []
    
    for gene_idx, gene in enumerate(genes):
        true_seq = targets[:, gene_idx]
        pred_seq = predictions[:, gene_idx]
        
        # Time-lagged correlation
        temp_corr = time_lagged_correlation(true_seq, pred_seq)
        temporal_metrics.append(temp_corr)
        
        # DTW distance
        dtw_dist = dtw_distance(true_seq, pred_seq)
        dtw_distances.append(dtw_dist)
        
        # Direction of changes
        true_changes = np.diff(true_seq)
        pred_changes = np.diff(pred_seq)
        dir_acc = np.mean(np.sign(true_changes) == np.sign(pred_changes))
        direction_accuracies.append(dir_acc)
    
    metrics['Mean_Temporal_Correlation'] = np.mean(temporal_metrics)
    metrics['Mean_DTW_Distance'] = np.mean(dtw_distances)
    metrics['Mean_Direction_Accuracy'] = np.mean(direction_accuracies)
    
    # Calculate rate of change metrics
    true_changes = np.diff(targets, axis=0)
    pred_changes = np.diff(predictions, axis=0)
    
    metrics['Mean_True_Change'] = np.mean(np.abs(true_changes))
    metrics['Mean_Pred_Change'] = np.mean(np.abs(pred_changes))
    metrics['Change_Magnitude_Ratio'] = metrics['Mean_Pred_Change'] / metrics['Mean_True_Change']
    
    return metrics

def create_gene_temporal_plots(predictions, targets, dataset, save_dir):
    """Create temporal pattern plots for all genes across multiple pages."""
    genes = list(dataset.node_map.keys())
    genes_per_page = 15  # Show 15 genes per page (5x3 grid)
    num_genes = len(genes)
    num_pages = (num_genes + genes_per_page - 1) // genes_per_page
    
    for page in range(num_pages):
        plt.figure(figsize=(20, 15))
        start_idx = page * genes_per_page
        end_idx = min((page + 1) * genes_per_page, num_genes)
        current_genes = genes[start_idx:end_idx]
        
        for i, gene in enumerate(current_genes):
            plt.subplot(5, 3, i+1)
            gene_idx = dataset.node_map[gene]
            
            # Plot actual and predicted values
            plt.plot(targets[:, gene_idx], 'b-', label='Actual', marker='o')
            plt.plot(predictions[:, gene_idx], 'r--', label='Predicted', marker='s')
            
            # Calculate metrics for this gene
            corr, _ = pearsonr(targets[:, gene_idx], predictions[:, gene_idx])
            rmse = np.sqrt(mean_squared_error(targets[:, gene_idx], predictions[:, gene_idx]))
            
            # Calculate changes
            actual_changes = np.diff(targets[:, gene_idx])
            pred_changes = np.diff(predictions[:, gene_idx])
            direction_acc = np.mean(np.sign(actual_changes) == np.sign(pred_changes))
            
            plt.title(f'Gene: {gene}\nCorr: {corr:.3f}, RMSE: {rmse:.3f}\nDir Acc: {direction_acc:.3f}')
            plt.xlabel('Time Step')
            plt.ylabel('Expression')
            if i == 0:  # Only show legend for first plot
                plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/temporal_patterns_page_{page+1}.png')
        plt.close()

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
    
    # 3. Gene temporal patterns for all genes
    create_gene_temporal_plots(predictions, targets, dataset, save_dir)

def get_predictions_and_targets(model, val_sequences, val_labels):
    """Extract predictions and targets from validation data."""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for seq, label in zip(val_sequences, val_labels):
            x, target = process_batch(seq, label)
            output = model(x)
            
            # Take last time step for predictions
            output = output[:, :, -1:, :]
            
            # Convert to numpy and reshape
            pred = output.squeeze().cpu().numpy()  # Should be [52] or [1, 52]
            true = target.squeeze().cpu().numpy()  # Should be [52] or [1, 52]
            
            if len(pred.shape) == 1:
                pred = pred.reshape(1, -1)
            if len(true.shape) == 1:
                true = true.reshape(1, -1)
            
            all_predictions.append(pred)
            all_targets.append(true)

    predictions = np.vstack(all_predictions)  # Should be [8, 52]
    targets = np.vstack(all_targets)        # Should be [8, 52]
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")
    
    return predictions, targets

def analyze_gene_characteristics(dataset, predictions, targets):
    """Analyze relationship between gene properties and prediction performance"""
    genes = list(dataset.node_map.keys())
    
    # Calculate gene correlations
    gene_correlations = {}
    for gene in genes:
        gene_idx = dataset.node_map[gene]
        pred_gene = predictions[:, gene_idx]  # [time_points]
        true_gene = targets[:, gene_idx]
        corr, _ = pearsonr(pred_gene, true_gene)
        gene_correlations[gene] = corr
    
    # Collect gene properties
    gene_stats = {gene: {
        'degree': len(dataset.base_graph[gene]),
        'expression_range': None,
        'expression_std': None,
        'correlation': gene_correlations[gene]
    } for gene in genes}
    
    # Calculate expression statistics
    for gene in genes:
        all_expressions = []
        for t in dataset.time_points:
            gene1_expr = dataset.df[dataset.df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
            gene2_expr = dataset.df[dataset.df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
            expr_values = np.concatenate([gene1_expr, gene2_expr])
            all_expressions.extend(expr_values)
        
        all_expressions = np.array(all_expressions)
        gene_stats[gene].update({
            'expression_range': np.ptp(all_expressions),
            'expression_std': np.std(all_expressions)
        })
    
    # Create analysis plots
    plt.figure(figsize=(15, 10))
    
    # 1. Correlation vs Degree
    plt.subplot(2, 2, 1)
    degrees = [gene_stats[gene]['degree'] for gene in genes]
    correlations = [gene_stats[gene]['correlation'] for gene in genes]
    plt.scatter(degrees, correlations)
    plt.xlabel('Number of Interactions')
    plt.ylabel('Prediction Correlation')
    plt.title('Gene Connectivity vs Prediction Performance')
    
    # 2. Correlation vs Expression Range
    plt.subplot(2, 2, 2)
    ranges = [gene_stats[gene]['expression_range'] for gene in genes]
    plt.scatter(ranges, correlations)
    plt.xlabel('Expression Range')
    plt.ylabel('Prediction Correlation')
    plt.title('Expression Variability vs Prediction Performance')
    
    plt.subplot(2, 2, 3)
    plt.hist(correlations, bins=20)
    plt.xlabel('Correlation')
    plt.ylabel('Count')
    plt.title('Distribution of Gene Correlations')
    
    plt.tight_layout()
    plt.savefig('plottings_STGCN/gene_analysis.png')
    plt.close()
    
    print("\nGene Analysis Summary:")
    print("\nTop 5 Most Connected Genes:")
    sorted_by_degree = sorted(gene_stats.items(), key=lambda x: x[1]['degree'], reverse=True)[:5]
    for gene, stats in sorted_by_degree:
        print(f"{gene}: {stats['degree']} connections, correlation: {stats['correlation']:.4f}")
    
    print("\nTop 5 Most Variable Genes:")
    sorted_by_range = sorted(gene_stats.items(), key=lambda x: x[1]['expression_range'], reverse=True)[:5]
    for gene, stats in sorted_by_range:
        print(f"{gene}: range {stats['expression_range']:.4f}, correlation: {stats['correlation']:.4f}")
    
    print("\nTop 5 Best Predicted Genes:")
    sorted_by_corr = sorted(gene_stats.items(), key=lambda x: x[1]['correlation'], reverse=True)[:5]
    for gene, stats in sorted_by_corr:
        print(f"{gene}: correlation {stats['correlation']:.4f}, connections: {stats['degree']}")
    
    return gene_stats

def analyze_temporal_patterns(dataset, predictions, targets):
    time_points = dataset.time_points
    genes = list(dataset.node_map.keys())

    temporal_stats = {
        'prediction_lag': [],  # Time shift between predicted and actual peaks
        'pattern_complexity': [],  # Number of direction changes
        'prediction_accuracy': []  # Accuracy by time point
    }

    time_point_accuracy = []
    for t in range(len(predictions)):
        corr = pearsonr(predictions[t].flatten(), targets[t].flatten())[0]
        time_point_accuracy.append(corr)
    
    plt.figure(figsize=(15, 5))
    plt.plot(time_point_accuracy)
    plt.xlabel('Time Point')
    plt.ylabel('Prediction Accuracy')
    plt.title('Prediction Accuracy Over Time')
    plt.savefig(f'plottings_STGCN/pred_accuracy.png')

    print("\nTemporal Analysis:")
    print(f"Best predicted time point: {np.argmax(time_point_accuracy)}")
    print(f"Worst predicted time point: {np.argmin(time_point_accuracy)}")
    print(f"Mean accuracy: {np.mean(time_point_accuracy):.4f}")
    print(f"Std of accuracy: {np.std(time_point_accuracy):.4f}")
    
    return temporal_stats

class Args:
    def __init__(self):
        self.Kt = 2 # temporal kernel size
        self.Ks = 3  # spatial kernel size
        self.n_his = 4  # historical sequence length
        self.n_pred = 1
       
        self.blocks = [
            [32, 32, 32],    # Input block
            [32, 48, 48],    # Single ST block (since temporal dim reduces quickly)
            [48, 32, 1]      # Output block
        ]
        self.act_func = 'glu'
        self.graph_conv_type = 'cheb_graph_conv'
        self.enable_bias = True
        self.droprate = 0.140128

if __name__ == "__main__":
    dataset = TemporalGraphDataset(
        csv_file='mapped/enhanced_interactions.csv',
        embedding_dim=32,
        seq_len=3,
        pred_len=1
    )

    model, val_sequences, val_labels, train_losses, val_losses = train_stgcn(dataset, val_ratio=0.2, step=1)
    
    metrics = evaluate_model_performance(model, val_sequences, val_labels, dataset)

    print("\nModel Performance Summary:")
    print("\nOverall Metrics:")
    for metric, value in metrics['Overall'].items():
        print(f"{metric}: {value:.4f}")

    print("\nGene Performance:")
    print(f"Mean Gene Correlation: {metrics['Gene']['Mean_Correlation']:.4f}")
    print(f"Best Performing Genes: {', '.join(metrics['Gene']['Best_Genes'])}")

    print("\nTemporal Performance:")
    print(f"Time-lagged Correlation: {metrics['Temporal']['Mean_Temporal_Correlation']:.4f}")
    print(f"DTW Distance: {metrics['Temporal']['Mean_DTW_Distance']:.4f}")
    print(f"Direction Accuracy: {metrics['Temporal']['Mean_Direction_Accuracy']:.4f}")
    print(f"Change Magnitude Ratio: {metrics['Temporal']['Change_Magnitude_Ratio']:.4f}")

    predictions, targets = get_predictions_and_targets(model, val_sequences, val_labels)
    gene_stats = analyze_gene_characteristics(dataset, predictions, targets)
    temporal_stats = analyze_temporal_patterns(dataset, predictions, targets)