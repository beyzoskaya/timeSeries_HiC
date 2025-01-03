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

def clean_gene_name(gene_name):
    """Clean gene name by removing descriptions and extra information"""
    if pd.isna(gene_name):
        return gene_name
    return gene_name.split('(')[0].strip()

class TemporalGraphDataset:
    def __init__(self, csv_file, embedding_dim=32, seq_len=5, pred_len=1): # I change the seq_len to more lower value
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
                num_walks=100,
                workers=1,
                p=1.5,
                q=1.5,
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

            #for t, features in temporal_features.items():
            #    print(f"Time {t}: Feature shape {features.shape}")
            
            print(f"Sample Node2Vec embedding for a node: {temporal_features[self.time_points[0]][0, :5]}")

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
            
            #print(f"Sequence graphs: {[g.x.shape for g in seq_graphs]}")
            #print(f"Label graphs: {[g.x.shape for g in label_graphs]}")

            #for graph in seq_graphs[:1]:  # Check only the first sequence graph
            #    print(graph.edge_index)
            
            #print(f"Feature mean: {torch.cat([g.x for g in seq_graphs]).mean(dim=0)}")
            #print(f"Feature std: {torch.cat([g.x for g in seq_graphs]).std(dim=0)}")
            
            #print(f"Label graphs (sample): {[g.x.shape for g in label_graphs[:3]]}")

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
    """Process batch data for training."""
    x = torch.stack([g.x for g in seq])
    x = x.permute(2, 0, 1).unsqueeze(0)  # [1, channels, time_steps, nodes]
    
    target = torch.stack([g.x for g in label])
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

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    num_epochs = 60
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
            
            x, _ = process_batch(seq, label)
            #print(f"Shape of x inside training: {x.shape}") # --> [1, 32, 5, 52]
            #print(f"Shape of target inside training: {target.shape}") # --> [1, 32, 1, 52]
            output = model(x)
            #print(f"Shape of output: {output.shape}") # --> [1, 32, 5, 52]
            _, target = process_batch(seq, label)
            #target = target[:,:,-1:, :]
            #print(f"Shape of target: {target.shape}") # --> [32, 1, 52]
            target = target[:, :, -1:, :]  # Keep only the last timestep
            output = output[:, :, -1:, :]
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        epoch_interaction_loss = 0

        with torch.no_grad():
            for seq,label in zip(val_sequences, val_labels):
                x, _ = process_batch(seq, label)
                
                output = model(x)
                _, target = process_batch(seq, label)
                target = target[:, :, -1:, :]  
                output = output[:, :, -1:, :]
                #print(f"Shape of output in validation: {output.shape}") # --> [1, 32, 5, 52]
                #print(f"Shape of target in validation: {target.shape}") # --> [32, 1, 52]
                #target = target[:,:,-1:, :]
                val_loss = criterion(output, target)

                output_corr = calculate_correlation(output)
                #print(f"Shape of output corr: {output_corr.shape}") # [32, 32]
                target_corr = calculate_correlation(target)
                #print(f"Shape of target corr: {target_corr.shape}") # [32, 32]
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

def evaluate_model_performance(model, val_sequences, val_labels, dataset, save_dir='plottings_STGCN'):
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
        predictions, targets, dataset.time_points[-len(predictions):], save_dir
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
    plt.savefig(f'{save_dir}/gene_performance_distribution.png')
    plt.close()
    
    return metrics

def calculate_temporal_metrics(predictions, targets, time_points, save_dir):
    """Calculate temporal stability metrics."""
    metrics = {}
    save_dir='plottings_STGCN'

    # Calculate temporal correlations
    temporal_correlations = []
    for t in range(len(predictions)):
        pred = predictions[t].flatten()
        true = targets[t].flatten()
        mask = ~(np.isnan(pred) | np.isnan(true))
        if mask.any():
            corr, _ = pearsonr(pred[mask], true[mask])
            temporal_correlations.append(corr)
    
    metrics['Mean_Temporal_Correlation'] = np.nanmean(temporal_correlations)
    metrics['Temporal_Correlation_Std'] = np.nanstd(temporal_correlations)
    
    # Plot temporal stability
    plt.figure(figsize=(10, 6))
    plt.plot(temporal_correlations, marker='o')
    plt.xlabel('Time Step')
    plt.ylabel('Correlation')
    plt.title('Temporal Stability of Predictions')
    plt.grid(True)
    plt.savefig(f'{save_dir}/temporal_stability.png')
    plt.close()
    
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
    save_dir='evaluation_results'
    )

    print("\nModel Performance Summary:")
    print(f"Overall Pearson Correlation: {metrics['Overall']['Pearson_Correlation']:.4f}")
    print(f"RMSE: {metrics['Overall']['RMSE']:.4f}")
    print(f"R² Score: {metrics['Overall']['R2_Score']:.4f}")
    print(f"\nGene-wise Performance:")
    print(f"Mean Gene Correlation: {metrics['Gene_Performance']['Mean_Correlation']:.4f}")
    print(f"Best Performing Genes: {', '.join(metrics['Gene_Performance']['Best_Genes'])}")
    print(f"\nTemporal Stability:")
    print(f"Mean Temporal Correlation: {metrics['Temporal_Stability']['Mean_Temporal_Correlation']:.4f}")