from create_graph_and_embeddings_STGCN import *
from STGCN_training import *
from evaluation import *
from utils import process_batch, calculate_correlation
from ontology_analysis import analyze_expression_levels, analyze_expression_levels_kmeans

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def filter_dataset_by_cluster(dataset, cluster_genes):
    """Create a new dataset containing only genes from the specified cluster."""
    filtered_dataset = deepcopy(dataset)
    
    # Update node mapping for cluster genes
    filtered_node_map = {gene: idx for idx, gene in enumerate(cluster_genes)}
    filtered_dataset.node_map = filtered_node_map
    filtered_dataset.num_nodes = len(cluster_genes)
    
    # Filter node features
    filtered_features = {}
    for t in dataset.time_points:
        indices = [dataset.node_map[gene] for gene in cluster_genes]
        filtered_features[t] = dataset.node_features[t][indices]
    filtered_dataset.node_features = filtered_features
    
    # Filter base graph
    filtered_base_graph = {}
    for gene in cluster_genes:
        if gene in dataset.base_graph:
            neighbors = {}
            for neighbor, weight in dataset.base_graph[gene].items():
                if neighbor in cluster_genes:
                    # Ensure weight is a numerical value
                    if isinstance(weight, dict):
                        weight = float(weight.get('weight', 1.0))  # Default to 1.0 if no weight found
                    else:
                        weight = float(weight)
                    neighbors[neighbor] = weight
            filtered_base_graph[gene] = neighbors
    filtered_dataset.base_graph = filtered_base_graph
    
    # Create edge index and weights
    edge_index = []
    edge_weights = []
    
    for source_gene, neighbors in filtered_base_graph.items():
        if source_gene not in filtered_node_map:
            continue
        source_idx = filtered_node_map[source_gene]
        
        for neighbor_gene, weight in neighbors.items():
            # Only add edge if both genes are in the cluster
            if neighbor_gene in filtered_node_map:
                neighbor_idx = filtered_node_map[neighbor_gene]
                # Add edges in both directions (undirected graph)
                edge_index.extend([[source_idx, neighbor_idx], [neighbor_idx, source_idx]])
                edge_weights.extend([float(weight), float(weight)])
        
    print(f"\nFiltered graph statistics:")
    print(f"Number of edges: {len(edge_weights)//2}")  # Divide by 2 because we add each edge twice
    print(f"Number of nodes: {len(filtered_node_map)}")
    
    if edge_index:
        filtered_dataset.edge_index = torch.tensor(edge_index).t().contiguous()
        filtered_dataset.edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)
    else:
        # Handle case with no edges
        filtered_dataset.edge_index = torch.zeros((2, 0), dtype=torch.long)
        filtered_dataset.edge_attr = torch.zeros((0, 1), dtype=torch.float32)
    
    return filtered_dataset

def get_cluster_blocks(num_nodes, expression_level):
    if expression_level == 'high_expr':
        return [
            [32, 32, 32],
            [32, 48, 48],
            [48, 32, 1]
        ]
    elif expression_level == 'medium_expr':
        return [
            [32, 32, 32],
            [32, 24, 24],
            [24, 16, 1]
        ]
    else: # low_expr
        return [
            [32, 32, 32],
            [32, 16, 16],
            [16, 8, 1]
        ]
    
def train_cluster_models(dataset, temporal_loss_fn):
    clusters, gene_expressions = analyze_expression_levels_kmeans(dataset)

    cluster_models = {}
    cluster_metrics = {}

    for cluster_name, genes in clusters.items():
        print(f"\n=== Training {cluster_name} cluster model ===")
        print(f"Number of genes in cluster: {len(genes)}")

        cluster_dataset = filter_dataset_by_cluster(dataset, genes)

        args = Args()
        args.n_vertex = len(genes)

        args.blocks = get_cluster_blocks(len(genes), cluster_name)
        print(f"Model configuration for {cluster_name}:")
        print(f"Number of nodes: {args.n_vertex}")
        print(f"Block structure: {args.blocks}")

        try:
            # Train model
            model = STGCNGraphConvProjected(args, args.blocks, args.n_vertex)
            model = model.float()
            
            # Get sequences for training
            sequences, labels = cluster_dataset.get_temporal_sequences()
            
            if not sequences:
                print(f"No sequences generated for {cluster_name} cluster. Skipping...")
                continue
                
            # Split data
            n_samples = len(sequences)
            n_train = int(n_samples * (1 - 0.2))
            indices = torch.randperm(n_samples)
            train_idx = indices[:n_train]
            val_idx = indices[n_train:]
            train_sequences = [sequences[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            val_sequences = [sequences[i] for i in val_idx]
            val_labels = [labels[i] for i in val_idx]
            
            # Training setup
            optimizer = optim.Adam(model.parameters(),  lr=0.00005)
            #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
            best_val_loss = float('inf')
            patience_counter = 0
            criterion = nn.MSELoss()
            
            train_losses = []
            val_losses = []
            
            for epoch in range(100):
                # Training
                model.train()
                epoch_loss = 0
                
                for seq, label in zip(train_sequences, train_labels):
                    optimizer.zero_grad()
                    x, target = process_batch(seq, label)
                    output = model(x)
                    
                    loss = temporal_loss_fn(
                       output[:, :, -1:, :],
                       target,
                       x
                    )

                    #loss = criterion(output[:, :, -1:, :], target)
                    
                    if not torch.isnan(loss):
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        epoch_loss += loss.item()
                
                avg_train_loss = epoch_loss / len(train_sequences)
                train_losses.append(avg_train_loss)
                
                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for seq, label in zip(val_sequences, val_labels):
                        x, target = process_batch(seq, label)
                        output = model(x)
                        loss = temporal_loss_fn(output[:, :, -1:, :], target, x)
                        #loss = criterion(output[:, :, -1:, :], target)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_sequences)
                val_losses.append(avg_val_loss)
                
                # Learning rate scheduling
                #scheduler.step(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model = deepcopy(model.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= 10:
                        print(f"Early stopping at epoch {epoch}")
                        break
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
            
            model.load_state_dict(best_model)
            
            metrics = evaluate_model_performance(model, val_sequences, val_labels, cluster_dataset)
            
            cluster_models[cluster_name] = model
            cluster_metrics[cluster_name] = {
                'metrics': metrics,
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Train')
            plt.plot(val_losses, label='Validation')
            plt.title(f'Training Progress - {cluster_name}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(f'plottings_STGCN/training_curve_{cluster_name}.png')
            plt.close()
            
        except Exception as e:
            print(f"Error training {cluster_name} cluster: {str(e)}")
            continue
    
    return cluster_models, cluster_metrics

def test_cluster_predictions(cluster_models, dataset, temporal_loss_fn):
    """Test predictions for each cluster."""
    predictions = {}
    metrics = {}
    
    clusters, _ = analyze_expression_levels_kmeans(dataset)
    
    for cluster_name, model in cluster_models.items():
        cluster_dataset = filter_dataset_by_cluster(dataset, clusters[cluster_name])
        sequences, labels = cluster_dataset.get_temporal_sequences()
        
        model.eval()
        cluster_preds = []
        cluster_targets = []
        
        with torch.no_grad():
            for seq, label in zip(sequences, labels):
                x, target = process_batch(seq, label)
                output = model(x)
                
                pred = output[:, :, -1:, :].squeeze().cpu().numpy()
                true = target.squeeze().cpu().numpy()
                
                cluster_preds.append(pred)
                cluster_targets.append(true)
        
        predictions[cluster_name] = {
            'predictions': np.array(cluster_preds),
            'targets': np.array(cluster_targets)
        }
        
        # Calculate metrics
        mse = mean_squared_error(
            predictions[cluster_name]['targets'].flatten(),
            predictions[cluster_name]['predictions'].flatten()
        )
        
        metrics[cluster_name] = {
            'mse': mse,
            'rmse': np.sqrt(mse)
        }
    
    return predictions, metrics

class Args:
    def __init__(self):
        self.Kt = 3
        self.Ks = 3
        self.n_his = 3
        self.n_pred = 1
        self.act_func = 'glu'
        self.graph_conv_type = 'graph_conv'
        self.enable_bias = True
        self.droprate = 0.1
def create_gene_temporal_plots(predictions, targets, dataset, all_genes, clusters, save_dir):
    """Create temporal pattern plots for all genes across multiple pages."""
    genes_per_page = 15  # Show 15 genes per page (5x3 grid)
    num_genes = len(all_genes)
    num_pages = (num_genes + genes_per_page - 1) // genes_per_page
    
    # Create color mapping for clusters
    colors = {'high_expr': 'red', 'medium_expr': 'blue', 'low_expr': 'green'}
    
    for page in range(num_pages):
        plt.figure(figsize=(20, 15))
        start_idx = page * genes_per_page
        end_idx = min((page + 1) * genes_per_page, num_genes)
        current_genes = all_genes[start_idx:end_idx]
        
        for i, gene in enumerate(current_genes):
            plt.subplot(5, 3, i+1)
            gene_idx = all_genes.index(gene)
            
            # Get gene's cluster
            gene_cluster = next(cluster_name for cluster_name, genes in clusters.items() 
                              if gene in genes)
            
            # Plot actual and predicted values
            plt.plot(targets[:, gene_idx], 'b-', label='Actual', marker='o')
            plt.plot(predictions[:, gene_idx], 'r--', label='Predicted', marker='s')
            
            # Calculate metrics for this gene
            corr, _ = pearsonr(targets[:, gene_idx], predictions[:, gene_idx])
            rmse = np.sqrt(mean_squared_error(targets[:, gene_idx], predictions[:, gene_idx]))
            
            # Calculate direction accuracy
            actual_changes = np.diff(targets[:, gene_idx])
            pred_changes = np.diff(predictions[:, gene_idx])
            direction_acc = np.mean(np.sign(actual_changes) == np.sign(pred_changes))
            
            plt.title(f'Gene: {gene} ({gene_cluster})\nCorr: {corr:.3f}, RMSE: {rmse:.3f}\nDir Acc: {direction_acc:.3f}')
            plt.xlabel('Time Step')
            plt.ylabel('Expression')
            if i == 0:
                plt.legend()
            
            # Add grid for better readability
            plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/temporal_patterns_page_{page+1}.png')
        plt.close()
        print(f"Created page {page+1} showing genes {start_idx+1}-{end_idx}")

def combine_cluster_evaluations(cluster_models, dataset):
    """Evaluate the combined performance of all cluster models."""
    clusters, _ = analyze_expression_levels_kmeans(dataset)
    all_predictions = []
    all_targets = []
    all_genes = []
    
    # Process each cluster
    for cluster_name, model in cluster_models.items():
        print(f"\nEvaluating {cluster_name} cluster...")
        cluster_dataset = filter_dataset_by_cluster(dataset, clusters[cluster_name])
        
        # Get validation sequences
        sequences, labels = cluster_dataset.get_temporal_sequences()
        n_samples = len(sequences)
        n_val = int(n_samples * 0.2)  # 20% validation
        val_sequences = sequences[-n_val:]
        val_labels = labels[-n_val:]
        
        # Get predictions for validation set
        model.eval()
        with torch.no_grad():
            cluster_predictions = []
            cluster_targets = []
            
            for seq, label in zip(val_sequences, val_labels):
                x, target = process_batch(seq, label)
                output = model(x)
                
                # Take only the prediction point
                pred = output[:, :, -1:, :].squeeze().cpu().numpy()
                true = target.squeeze().cpu().numpy()
                
                if len(pred.shape) == 0:
                    pred = np.array([pred])
                if len(true.shape) == 0:
                    true = np.array([true])
                
                cluster_predictions.append(pred)
                cluster_targets.append(true)
            
            if cluster_predictions:
                cluster_predictions = np.concatenate([p.reshape(1, -1) if p.ndim == 1 else p for p in cluster_predictions])
                cluster_targets = np.concatenate([t.reshape(1, -1) if t.ndim == 1 else t for t in cluster_targets])
                
                all_predictions.append(cluster_predictions)
                all_targets.append(cluster_targets)
                all_genes.extend(clusters[cluster_name])
    
    # Combine predictions from all clusters
    pred_array = np.concatenate(all_predictions, axis=1)
    target_array = np.concatenate(all_targets, axis=1)
    
    print(f"\nFinal prediction array shape: {pred_array.shape}")
    print(f"Final target array shape: {target_array.shape}")
    
    # Calculate metrics
    overall_metrics = calculate_overall_metrics(pred_array, target_array)
    cluster_metrics = calculate_cluster_metrics(pred_array, target_array, all_genes, clusters)
    temporal_metrics = calculate_temporal_metrics(pred_array, target_array)
    
    # Create visualizations
    create_gene_temporal_plots(pred_array, target_array, 
                             dataset, all_genes, clusters, 
                             save_dir='plottings_STGCN')
    
    return {
        'overall': overall_metrics,
        'per_cluster': cluster_metrics,
        'temporal': temporal_metrics
    }

def calculate_overall_metrics(pred_array, target_array):
    """Calculate overall metrics across all predictions."""
    return {
        'MSE': mean_squared_error(target_array.flatten(), pred_array.flatten()),
        'MAE': mean_absolute_error(target_array.flatten(), pred_array.flatten()),
        'R2': r2_score(target_array.flatten(), pred_array.flatten()),
        'Pearson': pearsonr(target_array.flatten(), pred_array.flatten())[0]
    }

def calculate_cluster_metrics(pred_array, target_array, all_genes, clusters):
    """Calculate metrics for each cluster."""
    cluster_metrics = {}
    
    for cluster_name, cluster_genes in clusters.items():
        # Get indices for genes in this cluster
        cluster_indices = [i for i, gene in enumerate(all_genes) if gene in cluster_genes]
        if cluster_indices:
            # Select cluster predictions (all time points for cluster genes)
            cluster_pred = pred_array[:, cluster_indices].flatten()
            cluster_target = target_array[:, cluster_indices].flatten()
            
            cluster_metrics[cluster_name] = {
                'MSE': mean_squared_error(cluster_target, cluster_pred),
                'MAE': mean_absolute_error(cluster_target, cluster_pred),
                'R2': r2_score(cluster_target, cluster_pred),
                'Pearson': pearsonr(cluster_target, cluster_pred)[0]
            }
    
    return cluster_metrics

def calculate_temporal_metrics(pred_array, target_array):
    """Calculate temporal metrics across genes."""
    metrics = {}
    
    # Direction accuracy (for each gene across time)
    true_changes = np.diff(target_array, axis=1)
    pred_changes = np.diff(pred_array, axis=1)
    direction_match = np.sign(true_changes) == np.sign(pred_changes)
    metrics['Direction_Accuracy'] = np.mean(direction_match)
    
    # Temporal correlation for each gene
    temporal_corrs = []
    for i in range(pred_array.shape[0]):
        if len(pred_array[i]) >= 2:  # Need at least 2 points for correlation
            try:
                corr, _ = pearsonr(pred_array[i], target_array[i])
                if not np.isnan(corr):
                    temporal_corrs.append(corr)
            except Exception as e:
                continue
    
    metrics['Mean_Temporal_Correlation'] = np.mean(temporal_corrs) if temporal_corrs else np.nan
    
    # Change magnitude accuracy
    true_magnitude = np.abs(true_changes)
    pred_magnitude = np.abs(pred_changes)
    metrics['Change_Magnitude_Ratio'] = (np.mean(pred_magnitude) / np.mean(true_magnitude) 
                                       if np.mean(true_magnitude) != 0 else np.nan)
    
    print("\nTemporal Metrics Statistics:")
    print(f"Number of genes with valid correlations: {len(temporal_corrs)}")
    print(f"Average temporal correlation: {metrics['Mean_Temporal_Correlation']:.4f}")
    print(f"Direction accuracy: {metrics['Direction_Accuracy']:.4f}")
    print(f"Change magnitude ratio: {metrics['Change_Magnitude_Ratio']:.4f}")
    
    return metrics

def debug_temporal_sequences(dataset):
    """Debug function to understand temporal sequence creation"""
    print("\nDebugging Temporal Sequences:")
    print(f"Total time points: {len(dataset.time_points)}")
    print(f"Time points: {dataset.time_points}")
    print(f"Sequence length: {dataset.seq_len}")
    print(f"Prediction length: {dataset.pred_len}")
    
    sequences, labels = dataset.get_temporal_sequences()
    print(f"\nNumber of sequences created: {len(sequences)}")
    
    # Check first few sequences
    print("\nFirst 3 sequences structure:")
    for i in range(min(3, len(sequences))):
        seq = sequences[i]
        label = labels[i]
        print(f"\nSequence {i+1}:")
        times_used = dataset.time_points[i:i+dataset.seq_len]
        target_time = dataset.time_points[i+dataset.seq_len:i+dataset.seq_len+dataset.pred_len]
        print(f"Input times: {times_used}")
        print(f"Target time: {target_time}")
        
    return sequences, labels

def analyze_predictions(pred_array, target_array):
    """Analyze the structure of predictions and targets"""
    print("\nPrediction Array Analysis:")
    print(f"Prediction array shape: {pred_array.shape}")
    print(f"Target array shape: {target_array.shape}")
    print(f"Number of time points in predictions: {pred_array.shape[1]}")
    print("\nSample values from first gene:")
    print(f"First 5 predictions: {pred_array[0, :5]}")
    print(f"First 5 targets: {target_array[0, :5]}")

def main():
    dataset = TemporalGraphDataset(
    csv_file='mapped/enhanced_interactions.csv',
    embedding_dim=32,
    seq_len=3,
    pred_len=1
)
    debug_temporal_sequences(dataset)

    cluster_models, cluster_metrics = train_cluster_models(dataset, temporal_loss_for_projected_model)

    combined_metrics = combine_cluster_evaluations(cluster_models, dataset)
    
    print("\n=== Combined Model Performance ===")
    print("\nOverall Metrics:")
    for metric, value in combined_metrics['overall'].items():
        print(f"{metric}: {value:.4f}")
        
    print("\nCluster-specific Metrics:")
    for cluster, metrics in combined_metrics['per_cluster'].items():
        print(f"\n{cluster}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
            
    print("\nTemporal Metrics:")
    for metric, value in combined_metrics['temporal'].items():
        print(f"{metric}: {value:.4f}")
    for cluster_name, model in cluster_models.items():
        torch.save(model.state_dict(), f'plottings_STGCN/model_{cluster_name}.pth')

if __name__ == "__main__":
    main()
