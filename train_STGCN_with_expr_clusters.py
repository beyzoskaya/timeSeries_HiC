from create_graph_and_embeddings_STGCN import *
from STGCN_training import *
from evaluation import *
from utils import process_batch, calculate_correlation
from ontology_analysis import analyze_expression_levels

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
            [32, 16, 16],
            [16, 16, 1]
        ]
    elif expression_level == 'medium_expr':
        return [
            [32, 32, 32],
            [32, 16, 16],
            [16, 8, 1]
        ]
    else:
        return [
            [32, 32, 32],
            [32, 8, 8],
            [8, 4, 1]
        ]
    
def train_cluster_models(dataset, temporal_loss_fn):
    clusters, gene_expressions = analyze_expression_levels(dataset)

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
            n_train = int(n_samples * 0.8)  # 80-20 split
            indices = torch.randperm(n_samples)
            
            train_sequences = [sequences[i] for i in indices[:n_train]]
            train_labels = [labels[i] for i in indices[:n_train]]
            val_sequences = [sequences[i] for i in indices[n_train:]]
            val_labels = [labels[i] for i in indices[n_train:]]
            
            # Training setup
            optimizer = optim.Adam(model.parameters(),  lr=0.0001, weight_decay=1e-5)
            #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
            best_val_loss = float('inf')
            patience_counter = 0
            
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
    
    clusters, _ = analyze_expression_levels(dataset)
    
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

def combine_cluster_evaluations(cluster_models, dataset):
    """Evaluate the combined performance of all cluster models."""
    all_predictions = []
    all_targets = []
    cluster_specific_metrics = {}
    clusters, _ = analyze_expression_levels(dataset)
    
    # Get predictions from each cluster
    for cluster_name, model in cluster_models.items():
        print(f"\nEvaluating {cluster_name} cluster...")
        cluster_dataset = filter_dataset_by_cluster(dataset, clusters[cluster_name])
        sequences, labels = cluster_dataset.get_temporal_sequences()
        
        model.eval()
        with torch.no_grad():
            for seq, label in zip(sequences, labels):
                x, target = process_batch(seq, label)
                output = model(x)
                
                # Extract predictions and targets
                pred = output[:, :, -1:, :].squeeze().cpu().numpy()
                true = target.squeeze().cpu().numpy()
                
                # Store with cluster information
                for gene_idx, gene in enumerate(clusters[cluster_name]):
                    all_predictions.append({
                        'gene': gene,
                        'cluster': cluster_name,
                        'predictions': pred[:, gene_idx],
                        'original_idx': dataset.node_map[gene]
                    })
                    all_targets.append({
                        'gene': gene,
                        'cluster': cluster_name,
                        'targets': true[:, gene_idx],
                        'original_idx': dataset.node_map[gene]
                    })

    # Sort by original indices to maintain consistent order
    all_predictions.sort(key=lambda x: x['original_idx'])
    all_targets.sort(key=lambda x: x['original_idx'])
    
    # Convert to arrays for metric calculation
    pred_array = np.array([p['predictions'] for p in all_predictions])
    target_array = np.array([t['targets'] for t in all_targets])
    
    # Calculate overall metrics
    overall_metrics = {
        'MSE': mean_squared_error(target_array.flatten(), pred_array.flatten()),
        'MAE': mean_absolute_error(target_array.flatten(), pred_array.flatten()),
        'R2': r2_score(target_array.flatten(), pred_array.flatten()),
        'Pearson': pearsonr(target_array.flatten(), pred_array.flatten())[0]
    }
    
    # Calculate metrics per cluster
    cluster_metrics = {}
    for cluster_name in clusters.keys():
        cluster_indices = [i for i, p in enumerate(all_predictions) if p['cluster'] == cluster_name]
        if cluster_indices:
            cluster_pred = pred_array[cluster_indices].flatten()
            cluster_target = target_array[cluster_indices].flatten()
            cluster_metrics[cluster_name] = {
                'MSE': mean_squared_error(cluster_target, cluster_pred),
                'MAE': mean_absolute_error(cluster_target, cluster_pred),
                'R2': r2_score(cluster_target, cluster_pred),
                'Pearson': pearsonr(cluster_target, cluster_pred)[0]
            }
    
    # Calculate temporal metrics
    temporal_metrics = calculate_temporal_metrics(pred_array, target_array)
    
    # Create visualizations
    create_combined_evaluation_plots(pred_array, target_array, all_predictions, dataset)
    
    return {
        'overall': overall_metrics,
        'per_cluster': cluster_metrics,
        'temporal': temporal_metrics
    }

def calculate_temporal_metrics(predictions, targets):
    """Calculate temporal metrics for the combined predictions."""
    temporal_metrics = {}
    
    # Direction accuracy
    true_changes = np.diff(targets, axis=1)
    pred_changes = np.diff(predictions, axis=1)
    direction_match = np.sign(true_changes) == np.sign(pred_changes)
    temporal_metrics['Direction_Accuracy'] = np.mean(direction_match)
    
    # Temporal correlation
    temporal_corrs = []
    for i in range(predictions.shape[0]):
        corr, _ = pearsonr(predictions[i], targets[i])
        temporal_corrs.append(corr)
    temporal_metrics['Mean_Temporal_Correlation'] = np.mean(temporal_corrs)
    
    # Change magnitude accuracy
    true_magnitude = np.abs(true_changes)
    pred_magnitude = np.abs(pred_changes)
    temporal_metrics['Change_Magnitude_Ratio'] = np.mean(pred_magnitude) / np.mean(true_magnitude)
    
    return temporal_metrics

def create_combined_evaluation_plots(predictions, targets, all_predictions, dataset, save_dir='plottings_STGCN'):
    """Create visualization plots for the combined evaluation."""
    # 1. Overall scatter plot
    plt.figure(figsize=(10, 8))
    colors = {'high_expr': 'red', 'medium_expr': 'blue', 'low_expr': 'green'}
    for cluster in colors:
        mask = [p['cluster'] == cluster for p in all_predictions]
        if any(mask):
            cluster_preds = predictions[mask].flatten()
            cluster_targets = targets[mask].flatten()
            plt.scatter(cluster_targets, cluster_preds, 
                       alpha=0.5, label=cluster, c=colors[cluster])
    
    plt.plot([targets.min(), targets.max()], 
             [targets.min(), targets.max()], 'k--', label='Perfect Prediction')
    plt.xlabel('True Expression')
    plt.ylabel('Predicted Expression')
    plt.title('Expression Prediction Performance by Cluster')
    plt.legend()
    plt.savefig(f'{save_dir}/combined_prediction_scatter.png')
    plt.close()
    
    # 2. Temporal pattern plot for selected genes
    plt.figure(figsize=(15, 10))
    genes_per_cluster = 3
    for i, cluster in enumerate(['high_expr', 'medium_expr', 'low_expr']):
        genes = [p['gene'] for p in all_predictions if p['cluster'] == cluster][:genes_per_cluster]
        for j, gene in enumerate(genes):
            plt.subplot(3, genes_per_cluster, i*genes_per_cluster + j + 1)
            gene_idx = [idx for idx, p in enumerate(all_predictions) if p['gene'] == gene][0]
            plt.plot(targets[gene_idx], 'b-', label='Actual')
            plt.plot(predictions[gene_idx], 'r--', label='Predicted')
            plt.title(f'{gene}\n({cluster})')
            if i == 0 and j == 0:
                plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/combined_temporal_patterns.png')
    plt.close()

def print_combined_evaluation_results(metrics):
    """Print the combined evaluation results in a formatted way."""
    print("\n=== Overall Model Performance ===")
    print("\nOverall Metrics:")
    for metric, value in metrics['overall'].items():
        print(f"{metric}: {value:.4f}")
    
    print("\nPerformance by Cluster:")
    for cluster, cluster_metrics in metrics['per_cluster'].items():
        print(f"\n{cluster}:")
        for metric, value in cluster_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    print("\nTemporal Performance:")
    for metric, value in metrics['temporal'].items():
        print(f"{metric}: {value:.4f}")

def main():
    dataset = TemporalGraphDataset(
    csv_file='mapped/enhanced_interactions.csv',
    embedding_dim=32,
    seq_len=3,
    pred_len=1
)

    cluster_models, cluster_metrics = train_cluster_models(dataset, temporal_loss_for_projected_model)

    combined_metrics = combine_cluster_evaluations(cluster_models, dataset)

    print_combined_evaluation_results(combined_metrics)

    for cluster_name, model in cluster_models.items():
        torch.save(model.state_dict(), f'plottings_STGCN/model_{cluster_name}.pth')

if __name__ == "__main__":
    main()
