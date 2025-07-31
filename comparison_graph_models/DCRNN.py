import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric_temporal.nn.recurrent import DCRNN 
import networkx as nx
import matplotlib.pyplot as plt
import os
import seaborn as sns
from node2vec import Node2Vec
from scipy.stats import pearsonr,spearmanr
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.spatial.distance import cdist
from create_graph_and_embeddings_STGCN import *
import seaborn as sns
from sklearn.manifold import TSNE
from create_graph_and_embeddings_STGCN import TemporalGraphDataset
    
def process_batch_for_dcrnn(seq, label, device='cpu'):
    # DCRNN processes sequences step by step
    x_seq = [g.x.to(device) for g in seq]  # List of [num_nodes, features]
    
    # Target: extract expression values (last feature)
    target_features = torch.stack([g.x for g in label])  # [pred_len, num_nodes, features]
    target = target_features[:, :, -1]  # [pred_len, num_nodes] - expression values
    
    return x_seq, target.to(device), seq

def train_dcrnn(model, dataset, epochs=10, learning_rate=1e-3, val_ratio=0.2, device='cpu'):
    model.to(device)
    model.train()

    sequences, labels = dataset.get_temporal_sequences()

    for t in dataset.time_points:
        edge_index_shape = dataset.temporal_edge_indices[t].shape[1]  # num_edges
        edge_attr_shape = dataset.temporal_edge_attrs[t].shape[0]     # num_edges
        
        if edge_index_shape != edge_attr_shape:
            print(f"ERROR at time {t}: edge_index has {edge_index_shape} edges but edge_attr has {edge_attr_shape}")
        else:
            print(f"Time {t}: {edge_index_shape} edges with attributes ✓")

    train_idx = torch.tensor([22, 9, 32, 15, 0, 3, 8, 18, 14, 13, 38, 2, 7, 4, 23, 37, 27, 29, 35, 17, 19, 25, 6, 21, 12, 10, 16, 39, 24, 33, 11, 34])
    val_idx = torch.tensor([28, 20, 26, 31, 30, 36, 1, 5])

    train_sequences, train_labels, val_sequences, val_labels, _, _ = dataset.split_sequences_from_idx(sequences, labels, train_idx, val_idx)
    
    print(f"Total sequences: {len(sequences)}")
    print(f"Train sequences: {len(train_sequences)}")
    print(f"Validation sequences: {len(val_sequences)}")
   
    if len(train_sequences) > 0:
        x_sample, target_sample, _ = process_batch_for_dcrnn(train_sequences[0], train_labels[0], device)
        print(f"Input sequence length for DCRNN: {len(x_sample)}")  # Should be seq_len
        print(f"Input shape per time step: {x_sample[0].shape}")  # Should be [num_nodes, features]
        print(f"Target shape: {target_sample.shape}")  # Should be [pred_len, num_nodes]
    
    optimizer = optim.Adam(model.parameters(), lr=0.0007, weight_decay=1e-5)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for seq_graphs, label_graphs in zip(train_sequences, train_labels):
            optimizer.zero_grad()
            
            x_input, y_true, seq_graphs_list = process_batch_for_dcrnn(seq_graphs, label_graphs, device)
            
            dynamic_edge_indices = []
            for i, seq_graph in enumerate(seq_graphs_list):
                dynamic_edge_indices.append(seq_graph.edge_index.to(device))

            edge_index = dataset.static_edge_index.to(device)
            edge_weight = dataset.static_edge_attr.squeeze().to(device) if hasattr(dataset, 'static_edge_attr') else None
                
            print(f"Sequence length: {len(x_input)}")  # seq_len
            print(f"Number of edge indices: {len(dynamic_edge_indices)}")  # should match seq_len
            print(f"Input shape per timestep: {x_input[0].shape}")  # [num_nodes, features]
            print(f"Target shape: {y_true.shape}")  # [pred_len, num_nodes]
            
            # DCRNN forward pass - processes sequence step by step
            h = None  # Initialize hidden state
            for t, x_t in enumerate(x_input):
                h = model(x_t, edge_index, edge_weight, h)
                print(f"Time step {t}: hidden state shape: {h.shape}")
            
            # h now contains the final prediction [num_nodes, out_channels]
            # Reshape to match target format
            output = h.unsqueeze(0)  # Add batch dimension: [1, num_nodes, out_channels]
            if output.shape[-1] == 1:
                output = output.squeeze(-1)  # [1, num_nodes] if out_channels=1
            
            # Reshape target to match output
            if y_true.dim() == 2 and y_true.shape[0] == 1:  # [1, num_nodes]
                y_true_reshaped = y_true
            else:  # [pred_len, num_nodes] -> [1, num_nodes] (take first/last)
                y_true_reshaped = y_true[-1:, :]  # Take last prediction step
            
            print(f"Final output shape: {output.shape}")
            print(f"Final target shape: {y_true_reshaped.shape}")
            
            input_stack = torch.stack(x_input)  # [seq_len, num_nodes, features]
            input_for_loss = input_stack.permute(1, 2, 0).unsqueeze(0)  # [1, num_nodes, features, seq_len]

            if output.dim() == 2:  # [1, num_nodes]
                output_for_loss = output.unsqueeze(1)  # [1, 1, num_nodes]
            else:
                output_for_loss = output
                
            if y_true_reshaped.dim() == 2:  # [1, num_nodes]
                target_for_loss = y_true_reshaped.unsqueeze(1)  # [1, 1, num_nodes]
            else:
                target_for_loss = y_true_reshaped

            loss = miRNA_enhanced_temporal_loss_dcrnn(output_for_loss, target_for_loss, input_for_loss)
            
            loss = miRNA_enhanced_temporal_loss_dcrnn(output_for_loss, target_for_loss, input_for_loss)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            break
        
        avg_train_loss = epoch_loss / len(train_sequences)
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seq_graphs, label_graphs in zip(val_sequences, val_labels):
                x_input, y_true, seq_graphs_list = process_batch_for_dcrnn(seq_graphs, label_graphs, device)
                
                dynamic_edge_indices = []
                for seq_graph in seq_graphs_list:
                    dynamic_edge_indices.append(seq_graph.edge_index.to(device))
                
                h = None  
                for t, x_t in enumerate(x_input):
                    h = model(x_t, edge_index, edge_weight, h)
                
                output = h.unsqueeze(0)
                if output.shape[-1] == 1:
                    output = output.squeeze(-1)
                
                y_true_reshaped = y_true[-1:, :] if y_true.dim() == 2 and y_true.shape[0] > 1 else y_true
            
                input_stack = torch.stack(x_input)
                print(f"input stack: {input_stack}")
                input_for_loss = input_stack.permute(1, 2, 0).unsqueeze(0)  # [1, num_nodes, features, seq_len]
                print(f"input_for_loss shape: {input_for_loss.shape}")
                output_for_loss = output.unsqueeze(1) if output.dim() == 2 else output
                print(f"output for loss shape: {output_for_loss.shape}")
                target_for_loss = y_true_reshaped.unsqueeze(1) if y_true_reshaped.dim() == 2 else y_true_reshaped
                print(f"target for loss shape: {target_for_loss.shape}")
                
                loss = miRNA_enhanced_temporal_loss_dcrnn(output_for_loss, target_for_loss, input_for_loss)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_sequences)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
    
    return model, val_sequences, val_labels, train_losses, val_losses, train_sequences, train_labels

def evaluate_dcrnn_performance(model, val_sequences, val_labels, dataset, device='cpu', save_dir='plottings_DCRNN'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for seq, label in zip(val_sequences, val_labels):
            x_input, target, seq_graphs_list = process_batch_for_dcrnn(seq, label, device)
            
            # Use static edge index instead of dynamic ones
            edge_index = dataset.static_edge_index.to(device)
            
            print(f"Evaluating - Input length: {len(x_input)}")
            
            # DCRNN forward pass with static edges
            h = None
            for t, x_t in enumerate(x_input):
                h = model(x_t, edge_index, None, h)  # Use static edges, no weights
            
            # Extract predictions and targets
            pred = h.cpu().numpy()  # [num_nodes, out_channels]
            true = target.cpu().numpy()  # [pred_len, num_nodes]
            
            # Handle output dimensions
            if pred.ndim == 2 and pred.shape[1] == 1:
                pred = pred.squeeze(1)  # [num_nodes]
            elif pred.ndim == 2:
                pred = pred[:, -1]  # Take last feature if multiple outputs
            
            if true.ndim == 2:
                true = true[-1, :]  # Take last prediction step
            
            all_predictions.append(pred)
            all_targets.append(true)
    
    predictions = np.array(all_predictions)  # [time_points, nodes]
    targets = np.array(all_targets)      # [time_points, nodes]
    
    print(f"Final predictions shape: {predictions.shape}")
    print(f"Final targets shape: {targets.shape}")

    overall_metrics = calculate_overall_metrics(predictions, targets)
    gene_metrics = calculate_gene_metrics(predictions, targets, dataset)
    temporal_metrics = calculate_temporal_metrics_detailly(predictions, targets, dataset)

    create_evaluation_plots(predictions, targets, dataset, save_dir)
    
    metrics = {
        'Overall': overall_metrics,
        'Gene': gene_metrics,
        'Temporal': temporal_metrics
    }
    
    return metrics

def get_dcrnn_predictions_and_targets(model, val_sequences, val_labels, dataset, device='cpu'):
    model.eval()
    all_predictions = []
    all_targets = []
    
    # In get_dcrnn_predictions_and_targets:
    edge_index = dataset.static_edge_index.to(device)

    with torch.no_grad():
        for seq, label in zip(val_sequences, val_labels):
            x_input, target, seq_graphs_list = process_batch_for_dcrnn(seq, label, device)

            # DCRNN forward pass with static edges
            h = None
            for t, x_t in enumerate(x_input):
                h = model(x_t, edge_index, None, h)
                        
            # Convert to numpy and reshape
            pred = h.cpu().numpy()  # [num_nodes, out_channels]
            true = target.cpu().numpy()  # [pred_len, num_nodes]
            
            # Handle multi-dimensional outputs
            if pred.ndim == 2 and pred.shape[1] == 1:
                pred = pred.squeeze(1)  # [num_nodes]
            elif pred.ndim == 2:
                pred = pred[:, -1]  # Take last feature
                
            if true.ndim == 2:
                true = true[-1, :]  # Take last prediction step
            
            all_predictions.append(pred.reshape(1, -1))
            all_targets.append(true.reshape(1, -1))

    predictions = np.vstack(all_predictions)  # [time_points, num_nodes]
    targets = np.vstack(all_targets)        # [time_points, num_nodes]
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")
    
    return predictions, targets

def plot_dcrnn_predictions_train_val(model, train_sequences, train_labels, val_sequences, val_labels, dataset, device='cpu', save_dir='plottings_DCRNN', genes_per_page=12):
    """Plot DCRNN gene predictions for both training and validation data."""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # Process training data
    train_predictions = []
    train_targets = []
    
    edge_index = dataset.static_edge_index.to(device)
    
    with torch.no_grad():
        # Get training predictions
        for seq, label in zip(train_sequences, train_labels):
            x_input, target, seq_graphs_list = process_batch_for_dcrnn(seq, label, device)
            
            h = None
            for t, x_t in enumerate(x_input):
                h = model(x_t, edge_index, None, h)
            
            pred = h.cpu().numpy()
            true = target.cpu().numpy()
            
            if pred.ndim == 2 and pred.shape[1] == 1:
                pred = pred.squeeze(1)
            elif pred.ndim == 2:
                pred = pred[:, -1]
                
            if true.ndim == 2:
                true = true[-1, :]
                
            train_predictions.append(pred)
            train_targets.append(true)
        
        # Get validation predictions  
        val_predictions = []
        val_targets = []
        
        for seq, label in zip(val_sequences, val_labels):
            x_input, target, seq_graphs_list = process_batch_for_dcrnn(seq, label, device)
            
            h = None
            for t, x_t in enumerate(x_input):
                h = model(x_t, edge_index, None, h)
            
            pred = h.cpu().numpy()
            true = target.cpu().numpy()
            
            if pred.ndim == 2 and pred.shape[1] == 1:
                pred = pred.squeeze(1)
            elif pred.ndim == 2:
                pred = pred[:, -1]
                
            if true.ndim == 2:
                true = true[-1, :]
                
            val_predictions.append(pred)
            val_targets.append(true)
    
    # Convert to arrays
    train_predictions = np.array(train_predictions)  # [train_sequences, nodes]
    train_targets = np.array(train_targets)          # [train_sequences, nodes]
    val_predictions = np.array(val_predictions)      # [val_sequences, nodes]
    val_targets = np.array(val_targets)              # [val_sequences, nodes]
    
    # Plot
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
        
            # Plot training data
            train_time_points = range(len(train_predictions))
            plt.plot(train_time_points, train_targets[:, gene_idx], label='Train Actual', color='blue', marker='o')
            plt.plot(train_time_points, train_predictions[:, gene_idx], label='Train Predicted', color='red', linestyle='--', marker='x')
     
            # Plot validation data
            val_time_points = range(len(train_predictions), len(train_predictions) + len(val_predictions))
            plt.plot(val_time_points, val_targets[:, gene_idx], label='Val Actual', color='green', marker='o')
            plt.plot(val_time_points, val_predictions[:, gene_idx], label='Val Predicted', color='orange', linestyle='--', marker='x')
            
            plt.title(f'Gene: {gene_name}', fontsize=16)
            plt.xlabel('Time Points', fontsize=14)
            plt.ylabel('Expression Value', fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(loc='center left', bbox_to_anchor=(-0.3, 0.5), fontsize=12, frameon=False)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/gene_predictions_page_{page + 1}.pdf', dpi=900)
        plt.close()

def miRNA_enhanced_temporal_loss_dcrnn(output, target, input_sequence, alpha=0.3, beta=0.2, gamma=0.3, delta=0.2):

    mse_loss = F.mse_loss(output, target)
    l1_loss = F.l1_loss(output, target)
 
    # Extract input expressions - DCRNN format: [1, num_nodes, features, seq_len]
    # Take only the last feature (expression values) from the input
    input_expressions = input_sequence[:, :, -1:, :]  # [1, num_nodes, 1, seq_len]
    last_input = input_expressions[:, :, 0, -1].unsqueeze(1)  # [1, num_nodes, 1] - last time step, expression feature
   
    output_reshaped = output.squeeze(1) if output.dim() > 2 else output  # [1, num_nodes]
    target_reshaped = target.squeeze(1) if target.dim() > 2 else target  # [1, num_nodes]
    last_input_reshaped = last_input.squeeze(2)  # [1, num_nodes]

    # Direction loss - captures if predictions move in the correct direction
    true_change = target_reshaped - last_input_reshaped
    pred_change = output_reshaped - last_input_reshaped

    # Cosine similarity to measure directional agreement
    true_norm = F.normalize(true_change, p=2, dim=-1)
    pred_norm = F.normalize(pred_change, p=2, dim=-1)
    direction_cosine = torch.sum(true_norm * pred_norm, dim=-1)
    direction_loss = 1 - torch.mean(direction_cosine)
    
    # Scale direction loss
    scaled_direction_loss = direction_loss * 0.01
    
    def enhanced_trend_correlation(pred, target, sequence_expr):
        # sequence_expr: [1, num_nodes, 1, seq_len] -> [1, num_nodes, seq_len]
        sequence_expr_2d = sequence_expr.squeeze(2)  # [1, num_nodes, seq_len]
        
        # Concatenate along time dimension
        pred_trend = torch.cat([sequence_expr_2d, pred.unsqueeze(2)], dim=2)  # [1, num_nodes, seq_len+1]
        target_trend = torch.cat([sequence_expr_2d, target.unsqueeze(2)], dim=2)  # [1, num_nodes, seq_len+1]

        def correlation_loss(x, y):
            x_centered = x - x.mean(dim=2, keepdim=True)
            y_centered = y - y.mean(dim=2, keepdim=True)
            x_norm = torch.sqrt(torch.sum(x_centered**2, dim=2) + 1e-8)
            y_norm = torch.sqrt(torch.sum(y_centered**2, dim=2) + 1e-8)
            correlation = torch.sum(x_centered * y_centered, dim=2) / (x_norm * y_norm + 1e-8)
            return 1 - correlation.mean()
        
        corr_loss = correlation_loss(pred_trend, target_trend)
        smoothness_loss = torch.mean(torch.abs(torch.diff(pred_trend, dim=2)))
       
        return corr_loss + 0.15 * smoothness_loss
 
    temporal_loss = enhanced_trend_correlation(output_reshaped, target_reshaped, input_expressions)
    scaled_temporal_loss = temporal_loss * 0.1
    
    # Consistency loss
    consistency_loss = torch.mean(torch.abs(output_reshaped - last_input_reshaped))
    
    total_loss = (
        alpha * l1_loss +
        beta * scaled_direction_loss + 
        gamma * scaled_temporal_loss +
        delta * consistency_loss
    )

    print(f"\nDCRNN Loss Components:")
    print(f"L1 loss: {l1_loss.item():.4f}")
    print(f"Direction Loss: {direction_loss.item():.4f}")
    print(f"Temporal Loss: {temporal_loss.item():.4f}")
    print(f"Consistency Loss: {consistency_loss.item():.4f}")
   
    return total_loss

def calculate_overall_metrics(predictions, targets):
    metrics = {}
    
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    metrics['MSE'] = mean_squared_error(target_flat, pred_flat)
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    metrics['MAE'] = mean_absolute_error(target_flat, pred_flat)
    metrics['R2_Score'] = r2_score(target_flat, pred_flat)
    metrics['Pearson_Correlation'], _ = pearsonr(target_flat, pred_flat)
    
    return metrics

def calculate_gene_metrics(predictions, targets, dataset):
    metrics = {}
    genes = list(dataset.node_map.keys())
    
    gene_correlations = []
    gene_rmse = []
    gene_spearman_correlations = []

    for gene_idx, gene in enumerate(genes):
        pred_gene = predictions[:, gene_idx]
        true_gene = targets[:, gene_idx]
        
        if np.std(pred_gene) == 0 or np.std(true_gene) == 0:
            corr = 0.0
            spearman_corr = 0.0
        else:
            try:
                corr, _ = pearsonr(pred_gene, true_gene)
                spearman_corr, spearman_p = spearmanr(pred_gene, true_gene)
            except:
                corr = 0.0
                spearman_corr = 0.0
        
        rmse = np.sqrt(mean_squared_error(true_gene, pred_gene))
        
        gene_correlations.append((gene, corr))
        gene_spearman_correlations.append((gene, spearman_corr))
        gene_rmse.append(rmse)
    
    gene_correlations.sort(key=lambda x: x[1], reverse=True)
    
    metrics['Mean_Correlation'] = np.mean([corr for _, corr in gene_correlations])
    metrics['Mean_Spearman_Correlation'] = np.mean([corr for _, corr in gene_spearman_correlations])
    metrics['Best_Genes_Pearson'] = [gene for gene, _ in gene_correlations[:5]]
    metrics['Best_Genes_Spearman'] = [gene for gene, _ in gene_spearman_correlations[:5]]
    metrics['Gene_RMSE'] = {gene: rmse for gene, rmse in zip(genes, gene_rmse)}
    
    return metrics

def calculate_temporal_metrics_detailly(predictions, targets, dataset):
    metrics = {}
    
    def time_lagged_correlation(true_seq, pred_seq, max_lag=3):
        correlations = []
        for lag in range(max_lag + 1):
            if lag == 0:
                corr = np.corrcoef(true_seq, pred_seq)[0, 1]
            else:
                corr = np.corrcoef(true_seq[lag:], pred_seq[:-lag])[0, 1]
            correlations.append(corr)
        return np.max(correlations)
    
    def dtw_distance(true_seq, pred_seq):
        n, m = len(true_seq), len(pred_seq)
        dtw_matrix = np.zeros((n+1, m+1))
        dtw_matrix[1:, 0] = np.inf
        dtw_matrix[0, 1:] = np.inf
        
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = abs(true_seq[i-1] - pred_seq[j-1])
                dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],
                                            dtw_matrix[i, j-1],
                                            dtw_matrix[i-1, j-1])
        return dtw_matrix[n, m]
    
    genes = list(dataset.node_map.keys())
    temporal_metrics = []
    dtw_distances = []
    direction_accuracies = []
    
    for gene_idx, gene in enumerate(genes):
        true_seq = targets[:, gene_idx]
        pred_seq = predictions[:, gene_idx]
        
        temp_corr = time_lagged_correlation(true_seq, pred_seq)
        temporal_metrics.append(temp_corr)
        
        dtw_dist = dtw_distance(true_seq, pred_seq)
        dtw_distances.append(dtw_dist)
        
        true_changes = np.diff(true_seq)
        pred_changes = np.diff(pred_seq)
        dir_acc = np.mean(np.sign(true_changes) == np.sign(pred_changes))
        direction_accuracies.append(dir_acc)
    
    metrics['Mean_Temporal_Correlation'] = np.mean(temporal_metrics)
    metrics['Mean_DTW_Distance'] = np.mean(dtw_distances)
    metrics['Mean_Direction_Accuracy'] = np.mean(direction_accuracies)
    
    true_changes = np.diff(targets, axis=0)
    pred_changes = np.diff(predictions, axis=0)
    
    metrics['Mean_True_Change'] = np.mean(np.abs(true_changes))
    metrics['Mean_Pred_Change'] = np.mean(np.abs(pred_changes))
    metrics['Change_Magnitude_Ratio'] = metrics['Mean_Pred_Change'] / metrics['Mean_True_Change']
    
    return metrics

def create_evaluation_plots(predictions, targets, dataset, save_dir):
    plt.figure(figsize=(10, 8))
    plt.scatter(targets.flatten(), predictions.flatten(), alpha=0.1)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
    plt.xlabel('True Expression')
    plt.ylabel('Predicted Expression')
    plt.title('Expression Prediction Performance')
    plt.savefig(f'{save_dir}/overall_scatter.png')
    plt.close()
    
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

def analyze_gene_characteristics(dataset, predictions, targets):
    genes = list(dataset.node_map.keys())
    
    gene_correlations = {}
    for gene in genes:
        gene_idx = dataset.node_map[gene]
        pred_gene = predictions[:, gene_idx]
        true_gene = targets[:, gene_idx]
        corr, _ = pearsonr(pred_gene, true_gene)
        gene_correlations[gene] = corr
    
    gene_stats = {gene: {
        'degree': len(dataset.base_graph[gene]),
        'expression_range': None,
        'expression_std': None,
        'correlation': gene_correlations[gene]
    } for gene in genes}
    
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
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    degrees = [gene_stats[gene]['degree'] for gene in genes]
    correlations = [gene_stats[gene]['correlation'] for gene in genes]
    plt.scatter(degrees, correlations)
    plt.xlabel('Number of Interactions')
    plt.ylabel('Prediction Correlation')
    plt.title('Gene Connectivity vs Prediction Performance')
    
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
    plt.savefig('plottings_DCRNN/gene_analysis.png')
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
        'prediction_lag': [],
        'pattern_complexity': [],
        'prediction_accuracy': []
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
    plt.savefig(f'plottings_DCRNN/pred_accuracy.png')

    print("\nTemporal Analysis:")
    print(f"Best predicted time point: {np.argmax(time_point_accuracy)}")
    print(f"Worst predicted time point: {np.argmin(time_point_accuracy)}")
    print(f"Mean accuracy: {np.mean(time_point_accuracy):.4f}")
    print(f"Std of accuracy: {np.std(time_point_accuracy):.4f}")
    
    return temporal_stats


if __name__ == "__main__":
    dataset = TemporalGraphDataset(
        csv_file='/Users/beyzakaya/Desktop/timeSeries_HiC/mapped/mRNA/enhanced_interactions_synthetic_simple_mRNA.csv',
        embedding_dim=128,
        seq_len=3,
        pred_len=1
    )

    test_graph = dataset.get_pyg_graph(dataset.time_points[0])
    print(f"Test graph valid: {test_graph.validate()}")
    print(f"Has edge_attr: {hasattr(test_graph, 'edge_attr') and test_graph.edge_attr is not None}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DCRNN(
        in_channels=dataset.embedding_dim,
        out_channels=1,  # Predicting 1 value per node (expression)
        K=2,  # Diffusion steps
        bias=True
    ).to(device)
    
    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB") 

    print(f"DCRNN Model initialized with:")
    print(f"- Node features: {dataset.embedding_dim}")
    print(f"- Sequence length: {dataset.seq_len}")
    print(f"- Prediction length: {dataset.pred_len}")
    print(f"- Number of nodes: {dataset.num_nodes}")
    print(f"- Output channels: 1")
    print(f"- Diffusion steps (K): 2")
    print(f"- Device: {device}")
    
    trained_model, val_sequences, val_labels, train_losses, val_losses, train_sequences, train_labels = train_dcrnn(
        model=model,
        dataset=dataset,
        epochs=15,
        learning_rate=1e-4,
        val_ratio=0.2,
        device=device
    )
   
    metrics = evaluate_dcrnn_performance(trained_model, val_sequences, val_labels, dataset, device)
    
    plot_dcrnn_predictions_train_val(trained_model, train_sequences, train_labels, val_sequences, val_labels, dataset, device)

    print("\nDCRNN Model Performance Summary:")
    print("\nOverall Metrics:")
    for metric, value in metrics['Overall'].items():
        print(f"{metric}: {value:.4f}")

    print("\nGene Performance:")
    print(f"Mean Gene Correlation: {metrics['Gene']['Mean_Correlation']:.4f}")
    print(f"Mean Spearman Correlation: {metrics['Gene']['Mean_Spearman_Correlation']:.4f}")
    print(f"Best Performing Genes Pearson: {', '.join(metrics['Gene']['Best_Genes_Pearson'])}")
    print(f"Best Performing Genes Spearman: {', '.join(metrics['Gene']['Best_Genes_Spearman'])}")

    print("\nTemporal Performance:")
    print(f"Time-lagged Correlation: {metrics['Temporal']['Mean_Temporal_Correlation']:.4f}")
    print(f"DTW Distance: {metrics['Temporal']['Mean_DTW_Distance']:.4f}")
    print(f"Direction Accuracy: {metrics['Temporal']['Mean_Direction_Accuracy']:.4f}")
    print(f"Change Magnitude Ratio: {metrics['Temporal']['Change_Magnitude_Ratio']:.4f}")

    predictions, targets = get_dcrnn_predictions_and_targets(model, val_sequences, val_labels, dataset, device)
    gene_stats = analyze_gene_characteristics(dataset, predictions, targets)
    temporal_stats = analyze_temporal_patterns(dataset, predictions, targets)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('DCRNN Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(np.log(train_losses), label='Log Training Loss')
    plt.plot(np.log(val_losses), label='Log Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.title('DCRNN Training and Validation Loss (Log Scale)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('plottings_DCRNN/training_losses.png')
    plt.close()
    
    print(f"\nTraining completed. Results saved in 'plottings_DCRNN' directory.")