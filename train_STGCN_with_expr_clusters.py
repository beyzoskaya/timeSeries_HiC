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
            [64, 64, 64],
            [64, 32, 32],
            [32, 16, 1]
        ]
    elif expression_level == 'medium_expr':
        return [
            [32, 32, 32],
            [32, 16, 16],
            [16, 8, 1]
        ]
    else:
        return [
            [16, 16, 16],
            [16, 8, 8],
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
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
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
                scheduler.step(avg_val_loss)
                
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

def main():
    dataset = TemporalGraphDataset(
    csv_file='mapped/enhanced_interactions.csv',
    embedding_dim=32,
    seq_len=3,
    pred_len=1
)

    cluster_models, cluster_metrics = train_cluster_models(dataset, temporal_loss_for_projected_model)

    predictions, metrics = test_cluster_predictions(cluster_models, dataset, temporal_loss_for_projected_model)

    for cluster_name, model in cluster_models.items():
        torch.save(model.state_dict(), f'plottings_STGCN/model_{cluster_name}.pth')

if __name__ == "__main__":
    main()
