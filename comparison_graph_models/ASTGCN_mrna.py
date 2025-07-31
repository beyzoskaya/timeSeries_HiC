import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric_temporal.nn.attention import ASTGCN 
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
from STGCN_losses import temporal_loss_for_projected_model, enhanced_temporal_loss, gene_specific_loss


class ASTGCNDataset:
    def __init__(self, csv_file, embedding_dim=32, seq_len=3, pred_len=1):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.embedding_dim = embedding_dim
        
        self.df = pd.read_csv(csv_file)
        self.df['Gene1_clean'] = self.df['Gene1']
        self.df['Gene2_clean'] = self.df['Gene2']
        self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed', case=False)]

        unique_genes = pd.concat([self.df['Gene1_clean'], self.df['Gene2_clean']]).unique()
        self.node_map = {gene: idx for idx, gene in enumerate(unique_genes)}
        self.num_nodes = len(self.node_map)
        
        self.time_cols = [col for col in self.df.columns if 'Time_' in col]
        self.time_points = sorted(list(set([float(col.split('_')[-1])  
                                          for col in self.time_cols if 'Gene1' in col])))
        print(f"Found time points: {self.time_points}")
        print(f"Len of time points: {len(self.time_points)}")
        self.time_points = [tp for tp in self.time_points if tp != 154.0]
        self.df = self.df.loc[:, ~self.df.columns.str.contains('Time_154.0', case=False)]
        
        self.base_graph = self.create_base_graph()
        
        self.node_features, self.temporal_edge_indices, self.temporal_edge_attrs = \
            self.create_temporal_node_features_with_node2vec()
        
        self.edge_index, self.edge_attr = self.get_edge_index_and_attr()

        self.static_edge_index = self.edge_index
        self.static_edge_attr = self.edge_attr
        #self.dynamic_edges = True  

    def create_base_graph(self):
        G = nx.Graph()
        G.add_nodes_from(self.node_map.keys())
        
        for _, row in self.df.iterrows():
            hic_weight = row['HiC_Interaction']
            compartment_sim = 1 if row['Gene1_Compartment'] == row['Gene2_Compartment'] else 0
            tad_dist = abs(row['Gene1_TAD_Boundary_Distance'] - row['Gene2_TAD_Boundary_Distance'])
            tad_sim = 1 / (1 + tad_dist)
            ins_sim = 1 / (1 + abs(row['Gene1_Insulation_Score'] - row['Gene2_Insulation_Score']))
            
            weight = (hic_weight * 0.4 + 
                     compartment_sim * 0.2 + 
                     tad_sim * 0.2 + 
                     ins_sim * 0.2)
             
            G.add_edge(row['Gene1_clean'], row['Gene2_clean'], weight=weight)
        
        return G

    def create_temporal_node_features_with_node2vec(self):
        temporal_node_features = {}
        temporal_edge_indices = {}
        temporal_edge_attrs = {}

        for t in self.time_points:
            G = nx.Graph()
            G.add_nodes_from(self.node_map.keys())
            
            edge_index = []
            edge_weights = []
            edges_added = 0  # Debug counter
            
            for _, row in self.df.iterrows():
                gene1 = row['Gene1_clean']
                gene2 = row['Gene2_clean']
                
                gene1_expr = row.get(f'Gene1_Time_{t}', 0.0)
                gene2_expr = row.get(f'Gene2_Time_{t}', 0.0)
                expr_sim = 1 / (1 + abs(gene1_expr - gene2_expr))
                
                hic_weight = row['HiC_Interaction'] if not pd.isna(row['HiC_Interaction']) else 0
                compartment_sim = 1 if row['Gene1_Compartment'] == row['Gene2_Compartment'] else 0
                tad_dist = abs(row['Gene1_TAD_Boundary_Distance'] - row['Gene2_TAD_Boundary_Distance'])
                tad_sim = 1 / (1 + tad_dist)
                ins_sim = 1 / (1 + abs(row['Gene1_Insulation_Score'] - row['Gene2_Insulation_Score']))
                
                weight = (hic_weight * 0.25 +
                        compartment_sim * 0.1 +
                        tad_sim * 0.1 +
                        ins_sim * 0.1 +
                        expr_sim * 0.45)
                
                print(f"Processing edge {gene1} - {gene2} at time {t}: weight={weight}")
                
                if weight > 15:  # Only add edges above threshold
                    G.add_edge(gene1, gene2, weight=weight)
                    
                    i, j = self.node_map[gene1], self.node_map[gene2]
                    edge_index.extend([[i, j], [j, i]])
                    edge_weights.extend([weight, weight])
            
            print(f"Time {t}: Added {edges_added} edges, Total edges in graph: {G.number_of_edges()}")
            
            # Node2Vec on this temporal graph
            node2vec = Node2Vec(
                G,
                dimensions=self.embedding_dim,
                walk_length=25,
                num_walks=75,
                p=1.0,
                q=1.0,
                workers=1,
                seed=42
            )
            model = node2vec.fit(window=5, min_count=1, batch_words=4)
            
            embeddings = []
            for gene in self.node_map.keys():
                embeddings.append(model.wv[str(gene)] if str(gene) in model.wv else np.zeros(self.embedding_dim))
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
            
            temporal_node_features[t] = embeddings
            temporal_edge_indices[t] = torch.tensor(edge_index).t().contiguous()
            temporal_edge_attrs[t] = torch.tensor(edge_weights, dtype=torch.float32)

            temporal_edge_indices[t] = torch.tensor(edge_index).t().contiguous()

            edge_weights_tensor = torch.tensor(edge_weights, dtype=torch.float32)
            if edge_weights_tensor.dim() == 1:
                edge_weights_tensor = edge_weights_tensor.unsqueeze(1)  # [num_edges] -> [num_edges, 1]

            temporal_edge_attrs[t] = edge_weights_tensor
        
        return temporal_node_features, temporal_edge_indices, temporal_edge_attrs
    
    def get_edge_index_and_attr(self):
        edge_index = []
        edge_weights = []
        for u, v, d in self.base_graph.edges(data=True):
            i, j = self.node_map[u], self.node_map[v]
            edge_index.extend([[i, j], [j, i]])
            edge_weights.extend([d['weight'], d['weight']])
        return (
            torch.tensor(edge_index).t().contiguous(),
            torch.tensor(edge_weights, dtype=torch.float32)
        )

    def get_pyg_graph(self, time_point):
        data = Data(
            x=self.node_features[time_point],
            edge_index=self.temporal_edge_indices[time_point],
            edge_attr=self.temporal_edge_attrs[time_point],
            num_nodes=self.num_nodes
        )
        
        # Debug prints
        print(f"Time {time_point}:")
        print(f"  x shape: {data.x.shape}")
        print(f"  edge_index shape: {data.edge_index.shape}")
        print(f"  edge_attr shape: {data.edge_attr.shape if data.edge_attr is not None else 'None'}")
        print(f"  edge_attr range: {data.edge_attr.min():.3f}-{data.edge_attr.max():.3f}" if data.edge_attr is not None else "No edge_attr")
        
        return data

    def get_temporal_sequences(self):
        sequences, labels = [], []
        for i in range(len(self.time_points) - self.seq_len - self.pred_len + 1):
            input_times = self.time_points[i:i+self.seq_len]
            target_times = self.time_points[i+self.seq_len:i+self.seq_len+self.pred_len]
            seq_graphs = [self.get_pyg_graph(t) for t in input_times]
            label_graphs = [self.get_pyg_graph(t) for t in target_times]
            sequences.append(seq_graphs)
            labels.append(label_graphs)
        return sequences, labels

    def split_sequences(self, sequences, labels, train_idx=None, val_idx=None):
        if train_idx is not None and val_idx is not None:
            print(f"Using provided train and validation indices.")
            train_sequences = [sequences[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            val_sequences = [sequences[i] for i in val_idx]
            val_labels = [labels[i] for i in val_idx]
            return train_sequences, train_labels, val_sequences, val_labels, train_idx, val_idx
        else:
            torch.manual_seed(42)
            n_samples = len(sequences)
            n_train = int(n_samples * 0.8)
            indices = torch.randperm(n_samples)
            train_idx, val_idx = indices[:n_train], indices[n_train:]
            train_sequences = [sequences[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            val_sequences = [sequences[i] for i in val_idx]
            val_labels = [labels[i] for i in val_idx]
            return train_sequences, train_labels, val_sequences, val_labels, train_idx, val_idx
    
def process_batch_for_astgcn(seq, label, device='cpu'):
    # Extract node features from sequence
    x_seq = torch.stack([g.x for g in seq])  # [seq_len, num_nodes, features]
    x_seq = x_seq.permute(1, 2, 0).unsqueeze(0)  # [1, num_nodes, features, seq_len]
    
    # Target: ASTGCN outputs [batch, num_nodes, pred_len], so we need to match this
    # Extract the expression values (last feature) from label graphs
    target_features = torch.stack([g.x for g in label])  # [pred_len, num_nodes, features]
    # Take only the last feature (expression value) and reshape properly
    target = target_features[:, :, -1].unsqueeze(0)  # [1, pred_len, num_nodes]
    target = target.permute(0, 2, 1)  # [1, num_nodes, pred_len]
    
    return x_seq.to(device), target.to(device), seq

def process_batch_for_astgcn_static(seq, label, device='cpu'):
 
    x_seq = torch.stack([g.x for g in seq])  # [seq_len, num_nodes, features]
    x_seq = x_seq.permute(1, 2, 0).unsqueeze(0)  # [1, num_nodes, features, seq_len]
    
    # Target: extract the last feature (expression value) from label
    target = torch.stack([g.x[:, -1] for g in label])  # [pred_len, num_nodes]
    target = target.permute(1, 0).unsqueeze(0)  # [1, num_nodes, pred_len]
    
    return x_seq.to(device), target.to(device)

def train_astgcn(model, dataset, epochs=10, learning_rate=1e-3, val_ratio=0.2, device='cpu'):

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
        #x_sample, target_sample = process_batch_for_astgcn(train_sequences[0], train_labels[0], device)
        x_sample, target_sample, _ = process_batch_for_astgcn(train_sequences[0], train_labels[0], device)
        print(f"Input shape for ASTGCN: {x_sample.shape}")  # Should be [1, num_nodes, features, seq_len]
        print(f"Target shape: {target_sample.shape}")  # Should be [1, num_nodes, pred_len]
    
    #criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0007, weight_decay=1e-5)
    
    train_losses = []
    val_losses = []
    
    # Get edge indices for ASTGCN (it expects static edge indices)
    edge_index = dataset.static_edge_index.to(device)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for seq_graphs, label_graphs in zip(train_sequences, train_labels):
            optimizer.zero_grad()
            
            x_input, y_true, seq_graphs_list = process_batch_for_astgcn(seq_graphs, label_graphs, device)
            
            # Prepare dynamic edge indices and edge attributes
            dynamic_edge_indices = []
            dynamic_edge_attrs = []
            for i, seq_graph in enumerate(seq_graphs_list):
                dynamic_edge_indices.append(seq_graph.edge_index.to(device))
                if hasattr(seq_graph, 'edge_attr') and seq_graph.edge_attr is not None:
                    edge_attr = seq_graph.edge_attr.to(device)
                    dynamic_edge_attrs.append(edge_attr)
                    print(f"Time step {i}: {edge_attr.shape[0]} edges, weight range: {edge_attr.min():.3f}-{edge_attr.max():.3f}")
                else:
                    print(f"Time step {i}: No edge attributes found!")
                    num_edges = seq_graph.edge_index.shape[1]
                    dynamic_edge_attrs.append(torch.ones(num_edges, 1).to(device))
            print(f"x_seq shape: {x_input.shape}")  # should be [1, num_nodes, features, seq_len] --> ([1, 49, 32, 3])
            print(f"target shape: {y_true.shape}")  # should be [1, num_nodes, pred_len] --> ([1, 49, 1])
            print(f"len(dynamic_edge_indices): {len(dynamic_edge_indices)}")  # == seq_len --> 3
            print(f"dynamic_edge_indices[0].shape: {dynamic_edge_indices[0].shape}") # ([2, 204])
            
            # Try forward pass with edge attributes, fallback to topology only
            try:
                output = model(x_input, dynamic_edge_indices, dynamic_edge_attrs)
            except:
                output = model(x_input, dynamic_edge_indices)
            
            #loss = criterion(output, y_true)
            loss = enhanced_temporal_loss(output, y_true, x_input)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_sequences)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seq_graphs, label_graphs in zip(val_sequences, val_labels):
                x_input, y_true, seq_graphs_list = process_batch_for_astgcn(seq_graphs, label_graphs, device)
                
                dynamic_edge_indices = []
                dynamic_edge_attrs = []
                for seq_graph in seq_graphs_list:
                    dynamic_edge_indices.append(seq_graph.edge_index.to(device))
                    if hasattr(seq_graph, 'edge_attr') and seq_graph.edge_attr is not None:
                        dynamic_edge_attrs.append(seq_graph.edge_attr.to(device))
                    else:
                        num_edges = seq_graph.edge_index.shape[1]
                        dynamic_edge_attrs.append(torch.ones(num_edges, 1).to(device))
                
                try:
                    output = model(x_input, dynamic_edge_indices, dynamic_edge_attrs)
                except:
                    output = model(x_input, dynamic_edge_indices)
                    
                #loss = criterion(output, y_true)
                loss = enhanced_temporal_loss(output, y_true, x_input)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_sequences)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
    
    return model, val_sequences, val_labels, train_losses, val_losses, train_sequences, train_labels

def evaluate_astgcn_performance(model, val_sequences, val_labels, dataset, device='cpu', save_dir='plottings_ASTGCN'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    edge_index = dataset.static_edge_index.to(device)
    
    with torch.no_grad():
        for seq, label in zip(val_sequences, val_labels):
            x_input, target, seq_graphs_list = process_batch_for_astgcn(seq, label, device)
            
            dynamic_edge_indices = []
            dynamic_edge_attrs = []
            for seq_graph in seq_graphs_list:
                dynamic_edge_indices.append(seq_graph.edge_index.to(device))
                if hasattr(seq_graph, 'edge_attr') and seq_graph.edge_attr is not None:
                    dynamic_edge_attrs.append(seq_graph.edge_attr.to(device))
                else:
                    num_edges = seq_graph.edge_index.shape[1]
                    dynamic_edge_attrs.append(torch.ones(num_edges, 1).to(device))
            
            try:
                print(f"Using dynamic edge indices and attributes for ASTGCN")
                print(f"Input shape: {x_input.shape}, Edge indices: {[e.shape for e in dynamic_edge_indices]}, Edge attrs: {[e.shape for e in dynamic_edge_attrs]}")
                output = model(x_input, dynamic_edge_indices, dynamic_edge_attrs)
            except:
                output = model(x_input, dynamic_edge_indices)
            
            # Extract predictions and targets
            # ASTGCN output: [1, num_nodes, pred_len]
            # We want: [num_nodes] for single time step prediction
            pred = output.squeeze().cpu().numpy()  # [num_nodes, pred_len] -> [num_nodes] if pred_len=1
            true = target.squeeze().cpu().numpy()  # [num_nodes, pred_len] -> [num_nodes] if pred_len=1
            
            if len(pred.shape) > 1:
                pred = pred[:, -1]  # Take last prediction if multiple time steps
            if len(true.shape) > 1:
                true = true[:, -1]  # Take last target if multiple time steps
            
            all_predictions.append(pred)
            all_targets.append(true)
    
    predictions = np.array(all_predictions)  # [time_points, nodes]
    targets = np.array(all_targets)      # [time_points, nodes]
    
    print(f"Final predictions shape: {predictions.shape}")
    print(f"Final targets shape: {targets.shape}")
    
    # Calculate metrics using existing functions
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

def get_astgcn_predictions_and_targets(model, val_sequences, val_labels, dataset, device='cpu'):
    model.eval()
    all_predictions = []
    all_targets = []
    
    edge_index = dataset.static_edge_index.to(device)
    
    with torch.no_grad():
        for seq, label in zip(val_sequences, val_labels):
            #x_input, target = process_batch_for_astgcn(seq, label, device)
            x_input, target, seq_graphs_list = process_batch_for_astgcn(seq, label, device)
            dynamic_edge_indices = []
            for seq_graph in seq_graphs_list:
                dynamic_edge_indices.append(seq_graph.edge_index.to(device))

            output = model(x_input, dynamic_edge_indices)
                        
            # Convert to numpy and reshape
            pred = output.squeeze().cpu().numpy()  # [num_nodes, pred_len]
            true = target.squeeze().cpu().numpy()  # [num_nodes, pred_len]
            
            # Handle multi-step predictions
            if len(pred.shape) > 1:
                pred = pred[:, -1]  # Take last prediction
            if len(true.shape) > 1:
                true = true[:, -1]  # Take last target
            
            all_predictions.append(pred.reshape(1, -1))
            all_targets.append(true.reshape(1, -1))

    predictions = np.vstack(all_predictions)  # [time_points, num_nodes]
    targets = np.vstack(all_targets)        # [time_points, num_nodes]
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")
    
    return predictions, targets

def plot_astgcn_predictions_train_val(model, train_sequences, train_labels, val_sequences, val_labels, dataset, device='cpu', save_dir='plottings_ASTGCN', genes_per_page=12):
    """Plot ASTGCN gene predictions for both training and validation data."""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    all_sequences = train_sequences + val_sequences
    all_labels = train_labels + val_labels
    
    num_genes = dataset.num_nodes
    edge_index = dataset.static_edge_index.to(device)
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for seq, label in zip(all_sequences, all_labels):
            #x_input, target = process_batch_for_astgcn(seq, label, device)
            x_input, target, seq_graphs_list = process_batch_for_astgcn(seq, label, device)
            output = model(x_input, edge_index)
            
            pred = output.squeeze().cpu().numpy()
            true = target.squeeze().cpu().numpy()
            
            # Handle multi-step predictions
            if len(pred.shape) > 1:
                pred = pred[:, -1]
            if len(true.shape) > 1:
                true = true[:, -1]
                
            all_predictions.append(pred)
            all_targets.append(true)
    
    predictions = np.array(all_predictions)  # [time_points, nodes]
    targets = np.array(all_targets)          # [time_points, nodes]
    
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
        
            train_time_points = range(len(train_labels))
            plt.plot(train_time_points, targets[:len(train_labels), gene_idx], label='Train Actual', color='blue', marker='o')
            plt.plot(train_time_points, predictions[:len(train_labels), gene_idx], label='Train Predicted', color='red', linestyle='--', marker='x')
     
            val_time_points = range(len(train_labels), len(train_labels) + len(val_labels))
            plt.plot(val_time_points, targets[len(train_labels):, gene_idx], label='Val Actual', color='green', marker='o')
            plt.plot(val_time_points, predictions[len(train_labels):, gene_idx], label='Val Predicted', color='orange', linestyle='--', marker='x')
            
            plt.title(f'Gene: {gene_name}', fontsize=16)
            plt.xlabel('Time Points', fontsize=14)
            plt.ylabel('Expression Value', fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(loc='center left', bbox_to_anchor=(-0.3, 0.5), fontsize=12, frameon=False)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/gene_predictions_page_{page + 1}.pdf', dpi=900)
        plt.close()

# Keep all the existing evaluation functions (calculate_overall_metrics, calculate_gene_metrics, etc.)
# They remain the same as in your original code

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
    gene_spearman_correlations = []

    for gene_idx, gene in enumerate(genes):
        pred_gene = predictions[:, gene_idx]  # All timepoints for this gene
        true_gene = targets[:, gene_idx]
        
        # Handle cases with no variation
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
    
    # Sort genes by correlation
    gene_correlations.sort(key=lambda x: x[1], reverse=True)
    
    metrics['Mean_Correlation'] = np.mean([corr for _, corr in gene_correlations])
    metrics['Mean_Spearman_Correlation'] = np.mean([corr for _, corr in gene_spearman_correlations])
    metrics['Best_Genes_Pearson'] = [gene for gene, _ in gene_correlations[:5]]
    metrics['Best_Genes_Spearman'] = [gene for gene, _ in gene_spearman_correlations[:5]]
    metrics['Gene_RMSE'] = {gene: rmse for gene, rmse in zip(genes, gene_rmse)}
    
    return metrics

def calculate_temporal_metrics_detailly(predictions, targets, dataset):
    """Calculate temporal prediction metrics with more appropriate temporal measures."""
    metrics = {}
    
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
    plt.savefig('plottings_ASTGCN/gene_analysis.png')
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
    plt.savefig(f'plottings_ASTGCN/pred_accuracy.png')

    print("\nTemporal Analysis:")
    print(f"Best predicted time point: {np.argmax(time_point_accuracy)}")
    print(f"Worst predicted time point: {np.argmin(time_point_accuracy)}")
    print(f"Mean accuracy: {np.mean(time_point_accuracy):.4f}")
    print(f"Std of accuracy: {np.std(time_point_accuracy):.4f}")
    
    return temporal_stats

def analyze_problematic_genes(dataset, problematic_genes):
    gene_stats = {}
    
    # Initialize group-wide trackers
    group_hic_weights = []
    group_expression = []
    
    for gene in problematic_genes:
        gene_idx = dataset.node_map.get(gene)
        if gene_idx is None:
            continue
            
        stats = {
            'connections': 0,
            'hic_weights': [],
            'compartment_matches': 0,
            'tad_distances': [],
            'expression_variance': []
        }
        
        # Collect HiC data
        for _, row in dataset.df.iterrows():
            if row['Gene1_clean'] == gene or row['Gene2_clean'] == gene:
                stats['connections'] += 1
                stats['hic_weights'].append(row['HiC_Interaction'])
                stats['compartment_matches'] += 1 if row['Gene1_Compartment'] == row['Gene2_Compartment'] else 0
                stats['tad_distances'].append(abs(row['Gene1_TAD_Boundary_Distance'] - row['Gene2_TAD_Boundary_Distance']))
        
        # Collect expression data
        expr_values = []
        for t in dataset.time_points:
            expr = dataset.df[dataset.df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
            if len(expr) > 0:
                expr_values.append(expr[0])
        stats['expression_variance'] = expr_values
        
        # Calculate per-gene averages
        stats['avg_hic_weight'] = np.mean(stats['hic_weights']) if stats['hic_weights'] else 0
        stats['expression_mean'] = np.mean(expr_values) if expr_values else 0
        stats['expression_std'] = np.std(expr_values) if expr_values else 0
        
        # Append to group-wide trackers
        group_hic_weights.extend(stats['hic_weights'])
        group_expression.extend(expr_values)
        
        gene_stats[gene] = stats
    
    # Calculate group-wide averages
    overall_stats = {
        'group_hic_avg': np.mean(group_hic_weights) if group_hic_weights else 0,
        'group_expr_avg': np.mean(group_expression) if group_expression else 0,
        'group_expr_std': np.std(group_expression) if group_expression else 0
    }
    
    return gene_stats, overall_stats


if __name__ == "__main__":
    # Initialize dataset
    dataset = TemporalGraphDataset(
        csv_file='/Users/beyzakaya/Desktop/timeSeries_HiC/mapped/mRNA/enhanced_interactions_synthetic_simple_mRNA.csv',
        embedding_dim=32,
        seq_len=3,
        pred_len=1
    )

    test_graph = dataset.get_pyg_graph(dataset.time_points[0])
    print(f"Test graph valid: {test_graph.validate()}")
    print(f"Has edge_attr: {hasattr(test_graph, 'edge_attr') and test_graph.edge_attr is not None}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ASTGCN(
    nb_block=2,
    in_channels=dataset.embedding_dim,
    K=3,
    nb_chev_filter=32,
    nb_time_filter=16,
    time_strides=1,
    num_for_predict=dataset.pred_len,
    len_input=dataset.seq_len,
    num_of_vertices=dataset.num_nodes,
    normalization=None,  
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

    print(f"ASTGCN Model initialized with:")
    print(f"- Node features: {dataset.embedding_dim}")
    print(f"- Sequence length: {dataset.seq_len}")
    print(f"- Prediction length: {dataset.pred_len}")
    print(f"- Number of nodes: {dataset.num_nodes}")
    print(f"- Device: {device}")
    
    # Train the model
    trained_model, val_sequences, val_labels, train_losses, val_losses, train_sequences, train_labels = train_astgcn(
        model=model,
        dataset=dataset,
        epochs=20,
        learning_rate=1e-4,
        val_ratio=0.2,
        device=device
    )
    
    # Evaluate model performance
    metrics = evaluate_astgcn_performance(trained_model, val_sequences, val_labels, dataset, device)
    
    # Plot predictions for train and validation
    plot_astgcn_predictions_train_val(trained_model, train_sequences, train_labels, val_sequences, val_labels, dataset, device)

    # Print performance summary
    print("\nASTGCN Model Performance Summary:")
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

    # Additional analysis
    predictions, targets = get_astgcn_predictions_and_targets(model, val_sequences, val_labels, dataset, device)
    gene_stats = analyze_gene_characteristics(dataset, predictions, targets)
    temporal_stats = analyze_temporal_patterns(dataset, predictions, targets)

    # Analyze specific gene groups
    problematic_genes = ['MCPT4', 'THTPA', 'PRIM2', 'GUCY1A2', 'MMP-3']
    gene_stats_prob, overall_stats_prob = analyze_problematic_genes(dataset, problematic_genes)

    print(f"\nWorst correlated genes:")
    print(f"Group HiC Average: {overall_stats_prob['group_hic_avg']:.4f}")
    print(f"Group Expression: {overall_stats_prob['group_expr_avg']:.4f} ± {overall_stats_prob['group_expr_std']:.4f}")
    
    print("*****************************************************************************************")

    best_correlated_genes = ['VIM', 'integrin subunit alpha 8', 'hprt', 'ADAMTSL2', 'TTF-1']
    gene_stats_best, overall_stats_best = analyze_problematic_genes(dataset, best_correlated_genes)

    print(f"Best correlated genes:")
    print(f"Group HiC Average: {overall_stats_best['group_hic_avg']:.4f}")
    print(f"Group Expression: {overall_stats_best['group_expr_avg']:.4f} ± {overall_stats_best['group_expr_std']:.4f}")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('ASTGCN Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(np.log(train_losses), label='Log Training Loss')
    plt.plot(np.log(val_losses), label='Log Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.title('ASTGCN Training and Validation Loss (Log Scale)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('plottings_ASTGCN/training_losses.png')
    plt.close()
    
    print(f"\nTraining completed. Results saved in 'plottings_ASTGCN' directory.")