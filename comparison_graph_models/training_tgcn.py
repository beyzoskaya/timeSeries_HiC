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
from scipy.stats import pearsonr,spearmanr
#from STGCN.model.models import *
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#sys.path.append('./STGCN')
from baseline_graph_models import BaselineGCN, BaselineTGCN
import argparse
import random
from scipy.spatial.distance import cdist
from create_graph_and_embeddings import *

def set_seed(seed=42):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed) 

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42) 

def process_batch(seq, label):
    """Process batch data for training."""
    # Input: Use full embeddings
    x = torch.stack([g.x for g in seq])  # [seq_len, num_nodes, features]
    x = x.permute(2, 0, 1).unsqueeze(0)  # [1, features, seq_len, nodes]
    
    # Target: Use only expression values
    target = torch.stack([g.x[:, -1] for g in label])  # [1, nodes] (expression values)
    target = target.unsqueeze(1).unsqueeze(0)  # [1, 1, 1, nodes]
    
    return x, target


def train_stgcn(dataset, val_ratio=0.2):

    embedding_dim = dataset.embedding_dim  # same as you passed when creating dataset
    hidden_dim = 16
    out_dim = 8
    n_vertex = dataset.num_nodes

    sequences, labels = dataset.get_temporal_sequences()
    print(f"\nCreated {len(sequences)} temporal sequences")

    edge_index = sequences[0][0].edge_index  # get once (same for all)
    print(f"Edge index shape: {edge_index.shape}")

    model = BaselineTGCN(
        in_channels=embedding_dim,
        hidden_channels=hidden_dim,
        out_channels=out_dim
    ).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()

    train_sequences, train_labels, val_sequences, val_labels, train_idx, val_idx = dataset.split_sequences(sequences, labels)

    print(f"Train sequences: {len(train_sequences)} | Val sequences: {len(val_sequences)}")

    save_dir = 'plottings_TGCN'
    os.makedirs(save_dir, exist_ok=True)
    with open(f'{save_dir}/split_indices.txt', 'w') as f:
        f.write("Train Indices:\n" + ", ".join(map(str, train_idx)) + "\n")
        f.write("Validation Indices:\n" + ", ".join(map(str, val_idx)) + "\n")

    num_epochs = 60
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for seq, label in zip(train_sequences, train_labels):
            optimizer.zero_grad()

            # Pass the sequence of Data objects directly to the model
            output = model(seq)  # output shape: [seq_len, n_nodes, out_dim]

            # Prepare target tensor from label sequence of Data objects
            target = torch.stack([g.x for g in label], dim=0)  # shape: [seq_len, n_nodes, embedding_dim]

            # Predict only the last timestep
            output_last = output[-1]  # [n_nodes, out_dim]
            target_last = target[-1, :, :out_dim]  # select matching output dims if needed

            loss = criterion(output_last, target_last)

            if torch.isnan(loss):
                print("⚠️ NaN loss detected! Skipping batch.")
                continue

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_sequences)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for seq, label in zip(val_sequences, val_labels):
                output = model(seq)

                target = torch.stack([g.x for g in label], dim=0)

                output_last = output[-1]
                target_last = target[-1, :, :out_dim]

                val_loss = criterion(output_last, target_last)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_sequences)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

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
                print("Early stopping triggered!")
                break

    checkpoint = torch.load(f'{save_dir}/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1}")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model, val_sequences, val_labels, train_losses, val_losses, train_sequences, train_labels, edge_index


def evaluate_model_performance(model, val_sequences, val_labels, dataset, edge_index, save_dir='plottings_TGCN'):

    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for i, (seq, label) in enumerate(zip(val_sequences, val_labels)):

            # Forward pass: pass list of PyG Data objects directly
            output = model(seq)  # shape: [seq_len, n_nodes, out_dim]

            # Prepare target tensor from label sequence of Data objects
            target = torch.stack([g.x for g in label], dim=0)  # shape: [seq_len, n_nodes, embedding_dim]

            # Select last timestep predictions and targets
            output_last = output[-1].cpu()  # [n_nodes, out_dim]
            target_last = target[-1, :, :output_last.shape[1]].cpu()  # match target dims to output dims

            # Convert to numpy for metrics
            output_np = output_last.numpy()
            target_np = target_last.numpy()

            print(f"Sample {i}: output shape {output_np.shape}, target shape {target_np.shape}")

            all_predictions.append(output_np)
            all_targets.append(target_np)

    predictions = np.array(all_predictions)  # [num_samples, n_nodes, out_dim]
    targets = np.array(all_targets)          # [num_samples, n_nodes, out_dim]

    overall_metrics = calculate_overall_metrics(predictions, targets)
    gene_metrics = calculate_gene_metrics(predictions, targets, dataset)

    create_evaluation_plots(predictions, targets, dataset, save_dir)

    metrics = {
        'Overall': overall_metrics,
        'Gene': gene_metrics
    }

    return metrics

def calculate_overall_metrics(predictions, targets):
    """Calculate overall expression prediction metrics."""
    metrics = {}

    # Flatten over samples, nodes, and features for global metrics
    pred_flat = predictions.reshape(-1)
    target_flat = targets.reshape(-1)
    
    metrics['MSE'] = mean_squared_error(target_flat, pred_flat)
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    metrics['MAE'] = mean_absolute_error(target_flat, pred_flat)
    metrics['R2_Score'] = r2_score(target_flat, pred_flat)
    metrics['Pearson_Correlation'], _ = pearsonr(target_flat, pred_flat)
    metrics['Spearman_Correlation'], _ = spearmanr(target_flat, pred_flat)
    
    return metrics

def calculate_gene_metrics(predictions, targets, dataset):
    """Calculate gene-specific metrics averaged over samples."""
    metrics = {}
    genes = list(dataset.node_map.keys())
    num_samples = predictions.shape[0]

    gene_correlations = []
    gene_rmse = []
    gene_spearman_correlations = []

    for gene_idx, gene in enumerate(genes):
        # Extract all samples for this gene (across samples and features)
        pred_gene = predictions[:, gene_idx, 0]  # Assuming single output feature per node
        true_gene = targets[:, gene_idx, 0]

        # Compute correlations and RMSE across all samples
        corr, _ = pearsonr(pred_gene, true_gene)
        spearman_corr, _ = spearmanr(pred_gene, true_gene)
        rmse = np.sqrt(mean_squared_error(true_gene, pred_gene))

        gene_correlations.append((gene, corr))
        gene_spearman_correlations.append((gene, spearman_corr))
        gene_rmse.append(rmse)

    # Sort genes by correlation descending
    gene_correlations.sort(key=lambda x: x[1], reverse=True)
    gene_spearman_correlations.sort(key=lambda x: x[1], reverse=True)

    metrics['Mean_Correlation'] = np.mean([corr for _, corr in gene_correlations])
    metrics['Mean_Spearman_Correlation'] = np.mean([corr for _, corr in gene_spearman_correlations])
    metrics['Best_Genes_Pearson'] = [gene for gene, _ in gene_correlations[:5]]
    metrics['Best_Genes_Spearman'] = [gene for gene, _ in gene_spearman_correlations[:5]]
    metrics['Gene_RMSE'] = {gene: rmse for gene, rmse in zip(genes, gene_rmse)}

    return metrics

def create_gene_temporal_plots(predictions, targets, dataset, save_dir):
    """Create plots of predictions vs targets across samples for each gene."""
    genes = list(dataset.node_map.keys())
    genes_per_page = 15
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

            # Plot over samples for this gene's output feature 0
            plt.plot(targets[:, gene_idx, 0], 'b-', label='Actual', marker='o')
            plt.plot(predictions[:, gene_idx, 0], 'r--', label='Predicted', marker='s')

            corr, _ = pearsonr(targets[:, gene_idx, 0], predictions[:, gene_idx, 0])
            rmse = np.sqrt(mean_squared_error(targets[:, gene_idx, 0], predictions[:, gene_idx, 0]))

            # Direction accuracy (sign agreement between target and prediction changes)
            actual_changes = np.diff(targets[:, gene_idx, 0])
            pred_changes = np.diff(predictions[:, gene_idx, 0])
            direction_acc = np.mean(np.sign(actual_changes) == np.sign(pred_changes))

            plt.title(f'Gene: {gene}\nCorr: {corr:.3f}, RMSE: {rmse:.3f}\nDir Acc: {direction_acc:.3f}')
            plt.xlabel('Sample')
            plt.ylabel('Expression')
            if i == 0:
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

def plot_gene_predictions_train_val(model, train_sequences, train_labels, val_sequences, val_labels, dataset, edge_index, save_dir='plottings_TGCN', genes_per_page=12):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    all_sequences = train_sequences + val_sequences
    all_labels = train_labels + val_labels

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for seq, label in zip(all_sequences, all_labels):
            # Prepare input
            x_in = torch.stack([g.x.T for g in seq], dim=0)  # [seq_len, embedding_dim, n_nodes]
            x_in = x_in.permute(0, 2, 1)  # [seq_len, n_nodes, embedding_dim]

            output = model(x_in, edge_index)  # output shape: [n_nodes, out_dim]
            output = output.squeeze().cpu().numpy()  # [n_nodes] if out_dim=1
            target = label.squeeze().cpu().numpy()   # assuming label shape [n_nodes]

            all_predictions.append(output)
            all_targets.append(target)

    predictions = np.array(all_predictions)  # [num_samples, num_genes]
    targets = np.array(all_targets)          # [num_samples, num_genes]

    gene_names = list(dataset.node_map.keys())
    num_genes = dataset.num_nodes
    num_pages = (num_genes + genes_per_page - 1) // genes_per_page

    for page in range(num_pages):
        plt.figure(figsize=(20, 15))
        start_idx = page * genes_per_page
        end_idx = min((page + 1) * genes_per_page, num_genes)
        page_genes = gene_names[start_idx:end_idx]

        for i, gene_name in enumerate(page_genes):
            gene_idx = start_idx + i
            rows = (genes_per_page + 1) // 2
            plt.subplot(rows, 2, i + 1)

            # Train indices
            train_idx = range(len(train_labels))
            # Validation indices
            val_idx = range(len(train_labels), len(train_labels) + len(val_labels))

            plt.plot(train_idx, targets[:len(train_labels), gene_idx], label='Train Actual', color='blue', marker='o')
            plt.plot(train_idx, predictions[:len(train_labels), gene_idx], label='Train Predicted', color='red', linestyle='--', marker='x')

            plt.plot(val_idx, targets[len(train_labels):, gene_idx], label='Val Actual', color='green', marker='o')
            plt.plot(val_idx, predictions[len(train_labels):, gene_idx], label='Val Predicted', color='orange', linestyle='--', marker='x')

            plt.title(f'Gene: {gene_name}', fontsize=16)
            plt.xlabel('Sample Index', fontsize=14)
            plt.ylabel('Expression Value', fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(loc='center left', bbox_to_anchor=(-0.3, 0.5), fontsize=12, frameon=False)

        plt.tight_layout()
        plt.savefig(f'{save_dir}/gene_predictions_page_{page + 1}.pdf', dpi=900)
        plt.close()


def plot_gene_predictions_train_val_proper_label(
    model, 
    train_sequences, 
    train_labels, 
    val_sequences, 
    val_labels, 
    dataset,
    save_dir='plottings_TGCN', 
    genes_per_page=12
):
   
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # Combine train and val data for plotting
    all_sequences = train_sequences + val_sequences
    all_labels = train_labels + val_labels
    num_genes = dataset.num_nodes
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for seq, label in zip(all_sequences, all_labels):
            # Forward pass: pass the sequence of Data objects directly
            output = model(seq)  # output shape: [seq_len, n_nodes, out_dim]
            
            # Prepare target tensor from label sequence
            target = torch.stack([g.x for g in label], dim=0)  # shape: [seq_len, n_nodes, embedding_dim]
            
            # Select the last timestep
            output_last = output[-1].squeeze().cpu().numpy()  # shape: [n_nodes] or [n_nodes, out_dim]
            target_last = target[-1, :, :output.shape[-1]].squeeze().cpu().numpy()
            
            all_predictions.append(output_last)
            all_targets.append(target_last)
    
    # Convert lists to arrays: shape [num_samples, num_genes]
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    
    gene_names = list(dataset.node_map.keys())
    num_pages = (num_genes + genes_per_page - 1) // genes_per_page

    for page in range(num_pages):
        plt.figure(figsize=(20, 15))  
        start_idx = page * genes_per_page
        end_idx = min((page + 1) * genes_per_page, num_genes)
        page_genes = gene_names[start_idx:end_idx]
        
        for i, gene_name in enumerate(page_genes):
            gene_idx = start_idx + i
            rows = (genes_per_page + 1) // 2  
            plt.subplot(rows, 2, i + 1) 
            
            # Plot train
            train_time_points = range(len(train_labels))
            plt.plot(train_time_points, targets[:len(train_labels), gene_idx], label='Train Actual', color='blue', marker='o')
            plt.plot(train_time_points, predictions[:len(train_labels), gene_idx], label='Train Predicted', color='red', linestyle='--', marker='x')
     
            # Plot val
            val_time_points = range(len(train_labels), len(train_labels) + len(val_labels))
            plt.plot(val_time_points, targets[len(train_labels):, gene_idx], label='Val Actual', color='green', marker='o')
            plt.plot(val_time_points, predictions[len(train_labels):, gene_idx], label='Val Predicted', color='orange', linestyle='--', marker='x')
            
            plt.title(f'Gene: {gene_name}', fontsize=16)
            plt.xlabel('Sample Index', fontsize=14)
            plt.ylabel('Expression Value', fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            # Move legend outside
            plt.legend(loc='center left', bbox_to_anchor=(-0.3, 0.5), fontsize=12, frameon=False)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/gene_predictions_page_{page + 1}.png', dpi=300)
        plt.close()

def get_predictions_and_targets(model, val_sequences, val_labels, edge_index):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for i, (seq, label) in enumerate(zip(val_sequences, val_labels)):
            x_in = torch.stack([g.x.T for g in seq], dim=0)
            x_in = x_in.permute(0, 2, 1)
            output = model(x_in, edge_index)
            print(f"Sample {i}: output shape {output.shape}, target shape {label.shape}")

            output = output.squeeze().cpu().numpy()
            target = label.squeeze().cpu().numpy()

            print(f"Sample {i}: after squeeze, output shape {output.shape}, target shape {target.shape}")

            if len(output.shape) == 1:
                pred = output.reshape(1, -1)
            else:
                pred = output
            if len(target.shape) == 1:
                true = target.reshape(1, -1)
            else:
                true = target

            all_predictions.append(pred)
            all_targets.append(true)

    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)

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
    plt.savefig('plottings_TGCN/gene_analysis.png')
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
    
    print(f"\nCorrelation values for all genes:")
    sorted_by_corr_all_genes = sorted(gene_stats.items(), key=lambda x: x[1]['correlation'], reverse=True)
    for gene, stats in sorted_by_corr_all_genes:
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
    plt.savefig(f'plottings_TGCN/pred_accuracy.png')

    print("\nTemporal Analysis:")
    print(f"Best predicted time point: {np.argmax(time_point_accuracy)}")
    print(f"Worst predicted time point: {np.argmin(time_point_accuracy)}")
    print(f"Mean accuracy: {np.mean(time_point_accuracy):.4f}")
    print(f"Std of accuracy: {np.std(time_point_accuracy):.4f}")
    
    return temporal_stats

class Args_miRNA:
    def __init__(self):
       
        self.Kt=3
        self.Ks=3
        self.n_his=10
        self.n_pred = 1
       
        self.blocks = [
             [64, 64, 64],    # Input block
             [64, 48, 48],    # Single ST block (since temporal dim reduces quickly)
             [48, 32, 1]      # Output block
        ]

        self.act_func = 'glu'
        self.graph_conv_type = 'cheb_graph_conv'
        self.enable_bias = True
        self.droprate = 0.1


if __name__ == "__main__":
    dataset = TemporalGraphDataset(
        csv_file = '/Users/beyzakaya/Desktop/timeSeries_HiC/mapped/mRNA/enhanced_interactions_synthetic_simple_mRNA.csv',
        embedding_dim=32,
        #seq_len=6,
        seq_len=4,
        pred_len=1
    )

    model, val_sequences, val_labels, train_losses, val_losses, train_sequences, train_labels, edge_index = train_stgcn(dataset, val_ratio=0.2)
    #model, val_sequences, val_labels, train_losses, val_losses, train_sequences, train_labels = train_stgcn_check_overfitting(dataset, val_ratio=0.2)
    metrics = evaluate_model_performance(model, val_sequences, val_labels, dataset, edge_index)
    plot_gene_predictions_train_val_proper_label(model, train_sequences, train_labels, val_sequences, val_labels, dataset)

    print("\nModel Performance Summary:")
    print("\nOverall Metrics:")
    for metric, value in metrics['Overall'].items():
        print(f"{metric}: {value:.4f}")

    print("\nGene Performance:")
    print(f"Mean Gene Correlation: {metrics['Gene']['Mean_Correlation']:.4f}")
    print(f"Mean Spearman Correlation: {metrics['Gene']['Mean_Spearman_Correlation']:.4f}")
    print(f"Best Performing Genes Pearson: {', '.join(metrics['Gene']['Best_Genes_Pearson'])}")
    print(f"Best Performing Genes Spearman: {', '.join(metrics['Gene']['Best_Genes_Spearman'])}")

    predictions, targets = get_predictions_and_targets(model, val_sequences, val_labels, edge_index)
    gene_stats = analyze_gene_characteristics(dataset, predictions, targets)
    temporal_stats = analyze_temporal_patterns(dataset, predictions, targets)

