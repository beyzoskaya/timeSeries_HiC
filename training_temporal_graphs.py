import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
import matplotlib.pyplot as plt
import os
import seaborn as sns
from node2vec import Node2Vec
from scipy.stats import pearsonr
from models import  AttentionSTGCN, STGCNModel,TGCNModel

def clean_gene_name(gene_name):
    """Clean gene name by removing descriptions and extra information"""
    if pd.isna(gene_name):
        return gene_name
    return gene_name.split('(')[0].strip()

################# Analyze Data ######################
def analyze_label_structure(val_sequences, val_labels):
    """Analyze the structure of sequences and labels"""
    print("\n=== Label Structure Analysis ===")
    
    # Look at first sequence and label
    seq = val_sequences[0]
    label = val_labels[0]
    
    print("\nSequence structure:")
    print(f"Number of graphs in sequence: {len(seq)}")
    print(f"First graph features shape: {seq[0].x.shape}")
    print(f"Edge index shape: {seq[0].edge_index.shape}")
    
    print("\nLabel structure:")
    print(f"Number of graphs in label: {len(label)}")
    print(f"Label features shape: {label[0].x.shape}")
    
    # Look at actual values
    print("\nExample values for first node:")
    print("Sequence feature values:")
    for i, g in enumerate(seq):
        print(f"Time step {i}: {g.x[0, :5]}")  # First node, first 5 features
    
    print("\nLabel feature values:")
    print(f"Target: {label[0].x[0, :5]}")  # First node, first 5 features
    
    return seq[0].x.shape, label[0].x.shape

def get_raw_predictions(model, val_sequences, val_labels):
    """Get predictions without any averaging"""
    model.eval()
    with torch.no_grad():
        # Get one sequence prediction
        seq = val_sequences[0]
        label = val_labels[0]
        
        # Get prediction
        pred = model(seq)
        target = label[0].x  # Direct access without mean
        
        print("\n=== Prediction Analysis ===")
        print(f"Raw prediction shape: {pred.shape}")
        print(f"Raw target shape: {target.shape}")
        
        print("\nFirst node values:")
        print(f"Predicted: {pred[0, :5]}")  # First node, first 5 features
        print(f"Target: {target[0, :5]}")   # First node, first 5 features
        
        return pred, target

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.6, beta=0.4):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
        self.huber = nn.HuberLoss(delta=1.0)
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        huber_loss = self.huber(pred, target)
        
        # Correlation loss to preserve patterns
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        vx = pred_flat - torch.mean(pred_flat)
        vy = target_flat - torch.mean(target_flat)
        
        corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-12)
        corr_loss = 1 - corr
        
        return self.alpha * (0.5 * mse_loss + 0.5 * huber_loss) + self.beta * corr_loss

class ExpressionSensitiveLoss(nn.Module):
    def __init__(self, trend_weight=0.3, magnitude_weight=0.3, correlation_weight=0.4):
        super(ExpressionSensitiveLoss, self).__init__()
        self.trend_weight = trend_weight
        self.magnitude_weight = magnitude_weight
        self.correlation_weight = correlation_weight
        
    def forward(self, pred, target):
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        trend_loss = F.mse_loss(torch.sign(pred_diff), torch.sign(target_diff))
      
        magnitude_loss = torch.mean(torch.abs(
            torch.log1p(torch.abs(pred)) - torch.log1p(torch.abs(target))
        ))
        
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        vx = pred_flat - torch.mean(pred_flat)
        vy = target_flat - torch.mean(target_flat)
        
        corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-12)
        correlation_loss = 1 - corr
        
        total_loss = (
            self.trend_weight * trend_loss + 
            self.magnitude_weight * magnitude_loss + 
            self.correlation_weight * correlation_loss
        )
        
        return total_loss
"""
number of nodes in the graph. 52
43 time points
"""
class TemporalGraphDataset:
    def __init__(self, csv_file, embedding_dim=64, seq_len=5, pred_len=1): # I change the seq_len to more lower value
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
    
    def create_temporal_node_features(self):
        """Create node features that incorporate temporal information using raw expression values"""
        temporal_features = {}
        
        # Create node2vec model using base graph
        node2vec = Node2Vec(
            self.base_graph,
            dimensions=self.embedding_dim // 2,  # Half dims for structural features
            walk_length=10,
            num_walks=80,
            workers=1,
            p=1,
            q=1,
            weight_key='weight'
        )
        
        # Get structural embeddings
        structural_model = node2vec.fit(window=10, min_count=1)
        structural_embeddings = {gene: torch.tensor(structural_model.wv[gene], dtype=torch.float32) 
                            for gene in self.node_map.keys()}
        
        # Print some statistics about structural embeddings
        print("\nStructural embedding statistics:")
        all_struct_values = torch.stack(list(structural_embeddings.values()))
        print(f"Structural mean: {all_struct_values.mean():.4f}")
        print(f"Structural std: {all_struct_values.std():.4f}")
        
        # Combine structural embeddings with raw temporal features
        for t in self.time_points:
            features = []
            for gene in self.node_map.keys():
                # Get raw expression values
                gene_expr = self.df[self.df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
                if len(gene_expr) == 0:
                    gene_expr = self.df[self.df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
                
                # Use raw expression value
                expr_value = gene_expr[0] if len(gene_expr) > 0 else 0.0
                
                # Optional: Print first few values to verify
                if t == self.time_points[0] and len(features) < 5:
                    print(f"Gene {gene} - Raw expression value: {expr_value:.4f}")
                
                # Convert to tensor and repeat to match structural embedding dimension
                expr_tensor = torch.tensor([expr_value], dtype=torch.float32)
                
                # Combine features
                combined_feature = torch.cat([
                    structural_embeddings[gene],
                    expr_tensor.repeat(self.embedding_dim // 2)  # Repeat to match dimensions
                ])
                features.append(combined_feature)
            
            temporal_features[t] = torch.stack(features)
            
            # Print statistics for first time point
            #if t == self.time_points[0]:
            #    print(f"\nFeature statistics for time {t}:")
            #    print(f"Mean: {temporal_features[t].mean():.4f}")
            #    print(f"Std: {temporal_features[t].std():.4f}")
            #    print(f"Min: {temporal_features[t].min():.4f}")
            #    print(f"Max: {temporal_features[t].max():.4f}")
        
        return temporal_features
    
    
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
                
                weight = (hic_weight * 0.3 + 
                        compartment_sim * 0.15 + 
                        tad_sim * 0.15 + 
                        ins_sim * 0.15 + 
                        expr_sim * 0.25)
                
                G.add_edge(row['Gene1_clean'], row['Gene2_clean'], weight=weight)
            
            # Create Node2Vec embeddings with exact embedding_dim dimensions
            node2vec = Node2Vec(
                G,
                dimensions=self.embedding_dim,  # Use full embedding dimension
                walk_length=10,
                num_walks=80,
                workers=1,
                p=1,
                q=1,
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

        """
        Created sequence for times: [16.0, 17.0, 18.0, 19.0, 20.0] with targets: [21.0]
        Created sequence for times: [17.0, 18.0, 19.0, 20.0, 21.0] with targets: [22.0]
        Created sequence for times: [18.0, 19.0, 20.0, 21.0, 22.0] with targets: [23.0]
        Created sequence for times: [19.0, 20.0, 21.0, 22.0, 23.0] with targets: [24.0]
        Created sequence for times: [20.0, 21.0, 22.0, 23.0, 24.0] with targets: [25.0]
        Created sequence for times: [21.0, 22.0, 23.0, 24.0, 25.0] with targets: [26.0]
        Created sequence for times: [22.0, 23.0, 24.0, 25.0, 26.0] with targets: [27.0]
        Created sequence for times: [23.0, 24.0, 25.0, 26.0, 27.0] with targets: [28.0]
        """

        return sequences, labels
    
    """
    Number of nodes: 52
    Feature shape: torch.Size([52, 33])
    Embedding dimension: 32
    Graph node features shape: torch.Size([52, 33])
    Graph edge index shape: torch.Size([2, 242])
    Graph edge attr shape: torch.Size([242, 1])
    """
def analyze_label_distribution(model, val_sequences, val_labels, dataset):
    """Analyze the distribution of label values"""
    print("\nAnalyzing label value distribution...")
    
    all_labels = []
    for label_seq in val_labels:
        #label_tensor = torch.stack([g.x for g in label_seq]).mean(dim=0)
        label_tensor = torch.stack([g.x for g in label_seq]).squeeze(dim=0)
        all_labels.append(label_tensor.numpy())
    
    all_labels = np.concatenate(all_labels, axis=0)
    
    print(f"Label statistics:")
    print(f"Mean: {np.mean(all_labels):.4f}")
    print(f"Std: {np.std(all_labels):.4f}")
    print(f"Min: {np.min(all_labels):.4f}")
    print(f"Max: {np.max(all_labels):.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.hist(all_labels.flatten(), bins=50)
    plt.title("Distribution of Label Values")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig(f'plottings_TGCNModel_one_graph/label_distribution.png')
    plt.close()

def split_temporal_sequences(sequences, labels, train_size=0.8):
    """Split sequences while maintaining temporal order"""
    split_index = int(len(sequences) * train_size)
    
    train_seq = sequences[:split_index]
    print(f" Length of train_seq: {len(train_seq)}")
    train_labels = labels[:split_index]
    test_seq = sequences[split_index:]
    print(f" Length of test_seq: {len(test_seq)}")
    test_labels = labels[split_index:]
    
    return train_seq, train_labels, test_seq, test_labels

def train_model_with_early_stopping_combined_loss(
    model, train_sequences, train_labels, val_sequences, val_labels, 
    num_epochs=100, learning_rate=0.0001, patience=10, delta=1.0, threshold=1e-4):
    
    save_dir = 'plottings_TGCNModel_one_graph'
    os.makedirs(save_dir, exist_ok=True)

    print("\nChecking data ranges before training:")
    for i, (seq, label) in enumerate(zip(train_sequences[:1], train_labels[:1])):
        print(f"\nSequence {i} stats:")
        seq_data = torch.stack([g.x for g in seq])
        label_data = torch.stack([g.x for g in label])
        print(f"Input range: [{seq_data.min():.4f}, {seq_data.max():.4f}]")
        print(f"Label range: [{label_data.min():.4f}, {label_data.max():.4f}]")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = CombinedLoss(alpha=0.6, beta=0.4)
    best_val_loss = float('inf')
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for seq, label in zip(train_sequences, train_labels):
            optimizer.zero_grad()
            output = model(seq)
            target = torch.stack([g.x for g in label]).squeeze(dim=0) # remove mean

            #if epoch % 5 == 0:
            #    print(f"\nTarget statistics:")
            #    print(f"Mean: {target.mean().item():.4f}")
            #    print(f"Std: {target.std().item():.4f}")
            #    print(f"Min: {target.min().item():.4f}")
            #    print(f"Max: {target.max().item():.4f}")
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seq, label in zip(val_sequences, val_labels):
                output = model(seq)
                #target = torch.stack([g.x for g in label]).mean(dim=0)
                target = torch.stack([g.x for g in label]).squeeze(dim=0)
                val_loss += criterion(output, target).item()
        
        train_loss = total_loss / len(train_sequences)
        val_loss = val_loss / len(val_sequences)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}\n')
        
        if val_loss < best_val_loss - threshold:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'{save_dir}/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    
    # Load best model
    model.load_state_dict(torch.load(f'{save_dir}/best_model.pth'))
    
    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_dir}/training_progress.png')
    plt.close()
    
    return train_losses, val_losses

def analyze_gene_predictions(model, val_sequences, val_labels, dataset, 
                           save_dir='plottings_TGCNModel_one_graph'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    with torch.no_grad():
        all_predictions = []
        all_targets = []
        
        for seq, label in zip(val_sequences, val_labels):
            pred = model(seq)
            #target = torch.stack([g.x for g in label]).mean(dim=0)
            target = torch.stack([g.x for g in label]).squeeze(dim=0)
            all_predictions.append(pred.numpy())
            all_targets.append(target.numpy())
        
        predictions = np.stack(all_predictions)
        targets = np.stack(all_targets)
        
        genes = list(dataset.node_map.keys())
        gene_metrics = {}
        
        for i, gene in enumerate(genes):
            gene_pred = predictions[:, i, :]
            gene_target = targets[:, i, :]
            
            corr = pearsonr(gene_pred.flatten(), gene_target.flatten())[0]
            mse = np.mean((gene_pred - gene_target) ** 2)
            mae = np.mean(np.abs(gene_pred - gene_target))
            
            gene_metrics[gene] = {
                'MSE': mse,
                'MAE': mae,
                'Correlation': corr
            }
            
            # Plot individual gene predictions
            plt.figure(figsize=(10, 6))
            plt.plot(gene_pred[0], label='Predicted', alpha=0.7)
            plt.plot(gene_target[0], label='Actual', alpha=0.7)
            plt.title(f'Time Series Prediction for {gene}')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, f'gene_{gene}_timeseries.png'))
            plt.close()
        
        # Save metrics and create summary plots
        metrics_df = pd.DataFrame.from_dict(gene_metrics, orient='index')
        metrics_df.to_csv(os.path.join(save_dir, 'gene_metrics.csv'))
        
        top_genes = metrics_df.nlargest(5, 'Correlation')
        bottom_genes = metrics_df.nsmallest(5, 'Correlation')
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 2)
        plt.bar(bottom_genes.index, bottom_genes['Correlation'])
        plt.title('Top 5 Worst Predicted Genes')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'gene_performance_comparison.png'))
        plt.close()
             
        avg_corr = np.mean([m['Correlation'] for m in gene_metrics.values()])
        print(f"\nAverage Pearson Correlation: {avg_corr:.4f}")
        
    return gene_metrics, avg_corr

def analyze_interactions(model, val_sequences, val_labels, dataset, save_dir='plottings_TGCNModel_one_graph'):
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        pred = model(val_sequences[0])
        #target = torch.stack([g.x for g in val_labels[0]]).mean(dim=0)
        target = torch.stack([g.x for g in val_labels[0]]).squeeze(dim=0)
        
        pred_np = pred.numpy()
        target_np = target.numpy()
        
        pred_interactions = np.corrcoef(pred_np)
        true_interactions = np.corrcoef(target_np)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        sns.heatmap(true_interactions, ax=ax1, cmap='coolwarm')
        ax1.set_title('True Interactions')
        
        sns.heatmap(pred_interactions, ax=ax2, cmap='coolwarm')
        ax2.set_title('Predicted Interactions')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'interaction_comparison.png'))
        plt.close()
        
        interaction_corr = np.corrcoef(true_interactions.flatten(), 
                                     pred_interactions.flatten())[0, 1]
        
        with open(os.path.join(save_dir, 'interaction_metrics.txt'), 'w') as f:
            f.write(f'Interaction Correlation: {interaction_corr:.4f}\n')
        
        return interaction_corr

def evaluate_model_performance(model, val_sequences, val_labels, dataset, split_index, save_dir='plottings_TGCNModel_one_graph'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    metrics = {}
   
    time_points = dataset.time_points
    seq_length = dataset.seq_len
    start_idx = split_index + seq_length
    prediction_times = time_points[start_idx:]
    
    print("\nValidation Time Information:")
    print(f"Original time points: {time_points}")
    print(f"Split index: {split_index}")
    print(f"Sequence length: {seq_length}")
    print(f"Start index for predictions: {start_idx}")
    print(f"Prediction times: {prediction_times}")

    with torch.no_grad():
        all_predictions = []
        all_targets = []
        
        for seq, label in zip(val_sequences, val_labels):
            pred = model(seq)
            #target = torch.stack([g.x for g in label]).mean(dim=0)
            target = torch.stack([g.x for g in label]).squeeze(dim=0)
            all_predictions.append(pred.numpy())
            all_targets.append(target.numpy())
        
        predictions = np.stack(all_predictions)
        targets = np.stack(all_targets)
        
        # 1. Overall Time Series Metrics
        metrics['Overall'] = {
            'MSE': np.mean((predictions - targets) ** 2),
            'MAE': np.mean(np.abs(predictions - targets)),
            'RMSE': np.sqrt(np.mean((predictions - targets) ** 2)),
            'Pearson_Correlation': pearsonr(predictions.flatten(), targets.flatten())[0],
            'R2_Score': 1 - np.sum((predictions - targets) ** 2) / np.sum((targets - np.mean(targets)) ** 2)
        }
        
        genes = list(dataset.node_map.keys())
        gene_correlations = []
        
        for i, gene in enumerate(genes):
            gene_pred = predictions[:, i, :]
            #print(f"gene_pred.shape: {gene_pred.shape}")
            gene_target = targets[:, i, :]
            #print(f"gene_target.shape: {gene_target.shape}")
            corr = pearsonr(gene_pred.flatten(), gene_target.flatten())[0]
            gene_correlations.append(corr)
            
            plt.figure(figsize=(10, 6))
            plt.plot(gene_target[0], label='Actual', alpha=0.7)
            plt.plot(gene_pred[0], label='Predicted', alpha=0.7)
            plt.title(f'{gene} Prediction (corr={corr:.3f})')
            plt.legend()
            plt.savefig(os.path.join(save_dir, f'{gene}_prediction.png'))
            plt.close()
        
        metrics['Gene_Performance'] = {
            'Mean_Correlation': np.mean(gene_correlations),
            'Median_Correlation': np.median(gene_correlations),
            'Std_Correlation': np.std(gene_correlations),
            'Best_Genes': [genes[i] for i in np.argsort(gene_correlations)[-5:]],
            'Worst_Genes': [genes[i] for i in np.argsort(gene_correlations)[:5]]
        }
        
        temporal_corrs = []
        for t in range(predictions.shape[1]):
            corr = pearsonr(predictions[:, t, :].flatten(), 
                          targets[:, t, :].flatten())[0]
            temporal_corrs.append(corr)
        
        metrics['Temporal_Stability'] = {
            'Mean_Temporal_Correlation': np.mean(temporal_corrs),
            'Std_Temporal_Correlation': np.std(temporal_corrs),
            'Min_Temporal_Correlation': np.min(temporal_corrs),
            'Max_Temporal_Correlation': np.max(temporal_corrs)
        }
        
        with open(os.path.join(save_dir, 'full_evaluation.txt'), 'w') as f:
            f.write("Model Evaluation Results\n\n")
            for category, category_metrics in metrics.items():
                f.write(f"\n{category}:\n")
                for metric, value in category_metrics.items():
                    if isinstance(value, (float, int)):
                        f.write(f"{metric}: {value:.4f}\n")
                    else:
                        f.write(f"{metric}: {value}\n")
        
        return metrics
    
def evaluate_model_with_direct_values(model, val_sequences, val_labels, dataset, save_dir='plottings_TGCN_direct'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # Target times for predictions
    target_times = [21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0]
    
    with torch.no_grad():
        all_predictions = []
        all_targets = []
        
        # Get predictions without averaging
        for seq, label in zip(val_sequences, val_labels):
            pred = model(seq)
            target = label[0].x  # Direct access to features
            all_predictions.append(pred.numpy())
            all_targets.append(target.numpy())
            
            # Print shapes for debugging
            print(f"\nPrediction shape: {pred.shape}")
            print(f"Target shape: {target.shape}")
        
        # Stack predictions and targets
        predictions = np.stack(all_predictions)
        targets = np.stack(all_targets)
        
        # Plot for each gene
        genes = list(dataset.node_map.keys())
        for i, gene in enumerate(genes):
            # Get gene's predictions and targets
            # Use first feature column for plotting
            gene_pred = predictions[:, i, 0]  # [num_sequences]
            gene_target = targets[:, i, 0]    # [num_sequences]
            
            plt.figure(figsize=(12, 6))
            
            # Plot with actual time points
            plt.plot(target_times, gene_target, 'bo-', label='Actual', markersize=8)
            plt.plot(target_times, gene_pred, 'ro--', label='Predicted', markersize=8)
            
            plt.title(f'Gene: {gene}')
            plt.xlabel('Time Points')
            plt.ylabel('Expression Value')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Add value labels
            for t, pred, actual in zip(target_times, gene_pred, gene_target):
                plt.annotate(f'{actual:.2f}', (t, actual),
                           textcoords="offset points", xytext=(0,10),
                           ha='center', color='blue')
                plt.annotate(f'{pred:.2f}', (t, pred),
                           textcoords="offset points", xytext=(0,-15),
                           ha='center', color='red')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{gene}_prediction.png'), dpi=300)
            plt.close()
            
            if i == 0:
                print(f"\nValues for gene {gene}:")
                print("Time points:", target_times)
                print("Predicted:", gene_pred)
                print("Actual:", gene_target)
    
def split_temporal_sequences(sequences, labels, train_size=0.8):
    """Split sequences while maintaining temporal order and return indices"""
    split_index = int(len(sequences) * train_size)
    
    print("\nSplit Information:")
    print(f"Total sequences: {len(sequences)}")
    print(f"Split index: {split_index}")
    print(f"Training sequences: {split_index}")
    print(f"Validation sequences: {len(sequences) - split_index}")

    train_seq = sequences[:split_index]
    train_labels = labels[:split_index]
    val_seq = sequences[split_index:]
    val_labels = labels[split_index:]

    return train_seq, train_labels, val_seq, val_labels, split_index

if __name__ == "__main__":
    dataset = TemporalGraphDataset(
        csv_file='mapped/enhanced_interactions.csv',
        embedding_dim=32,
        seq_len=5,
        pred_len=1
    )
    sequences, labels = dataset.get_temporal_sequences()
    
    train_seq, train_labels, val_seq, val_labels,split_index = split_temporal_sequences(sequences, labels, train_size=0.8)
    
    model = TGCNModel(
    num_nodes=dataset.num_nodes,
    in_channels=32,       # Must match your embedding_dim
    hidden_channels=64,
    out_channels=32
).float() # convert model to float because I got type error
    
    def convert_sequences_to_float32(sequences):
        converted_sequences = []
        for seq in sequences:
            converted_seq = []
            for graph in seq:
                converted_graph = Data(
                    x=graph.x.float(),
                    edge_index=graph.edge_index,
                    edge_attr=graph.edge_attr.float() if graph.edge_attr is not None else None,
                    num_nodes=graph.num_nodes
                )
                converted_seq.append(converted_graph)
            converted_sequences.append(converted_seq)
        return converted_sequences
    # converted all sequences to float also for consistency
    train_seq = convert_sequences_to_float32(train_seq) 
    train_labels = convert_sequences_to_float32(train_labels)
    val_seq = convert_sequences_to_float32(val_seq)
    val_labels = convert_sequences_to_float32(val_labels)

    analyze_label_distribution(model, val_seq, val_labels, dataset) # analyze labels before training!

    train_losses, val_losses = train_model_with_early_stopping_combined_loss(
        model, train_seq, train_labels, val_seq, val_labels
    )

    print("\nAnalyzing data structure...")
    seq_shape, label_shape = analyze_label_structure(val_seq, val_labels)
    
    print("\nGetting raw predictions...")
    pred, target = get_raw_predictions(model, val_seq, val_labels)

    print("\nEvaluating model with direct values...")
    evaluate_model_with_direct_values(model, val_seq, val_labels, dataset)

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('plottings_TGCNModel_one_graph/training_progress.png')

    #print("\nAnalyzing predictions...")
    #gene_metrics, avg_corr = analyze_gene_predictions(model, val_seq, val_labels, dataset)
    interaction_corr = analyze_interactions(model, val_seq, val_labels, dataset)
    
    #print("\nPrediction Summary:")
    #print(f"Average Gene Correlation: {np.mean([m['Correlation'] for m in gene_metrics.values()]):.4f}")
    print(f"Interaction Preservation: {interaction_corr:.4f}")

    metrics = evaluate_model_performance(model, val_seq, val_labels, dataset, split_index)
    print("\nModel Performance Summary:")
    print(f"Overall Pearson Correlation: {metrics['Overall']['Pearson_Correlation']:.4f}")
    print(f"RMSE: {metrics['Overall']['RMSE']:.4f}")
    print(f"R² Score: {metrics['Overall']['R2_Score']:.4f}")
    print(f"\nGene-wise Performance:")
    print(f"Mean Gene Correlation: {metrics['Gene_Performance']['Mean_Correlation']:.4f}")
    print(f"Best Performing Genes: {', '.join(metrics['Gene_Performance']['Best_Genes'])}")
    print(f"\nTemporal Stability:")
    print(f"Mean Temporal Correlation: {metrics['Temporal_Stability']['Mean_Temporal_Correlation']:.4f}")