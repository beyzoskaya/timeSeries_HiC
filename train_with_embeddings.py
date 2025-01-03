import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import seaborn as sns
from node2vec import Node2Vec
from scipy.stats import pearsonr
from models import BaseSTGCN, EnhancedSTGCN, AttentionSTGCN
import numpy as np

def clean_gene_name(gene_name):
    """Clean gene name by removing descriptions and extra information"""
    if pd.isna(gene_name):
        return gene_name
    return gene_name.split('(')[0].strip()

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.6, beta=0.4):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
        self.huber = nn.HuberLoss(delta=1.0)
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        
        # Huber loss for robustness to outliers
        huber_loss = self.huber(pred, target)
        
        # Correlation loss to preserve patterns
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        vx = pred_flat - torch.mean(pred_flat)
        vy = target_flat - torch.mean(target_flat)
        
        corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-12)
        corr_loss = 1 - corr
        
        return self.alpha * (0.5 * mse_loss + 0.5 * huber_loss) + self.beta * corr_loss
    
class TemporalGraphDataset:
    def __init__(self, csv_file, embedding_dim=64, seq_len=10, pred_len=1):
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.df = pd.read_csv(csv_file)
        self.df['Gene1_clean'] = self.df['Gene1'].apply(clean_gene_name)
        self.df['Gene2_clean'] = self.df['Gene2'].apply(clean_gene_name)
        
        self.embedding_dim = embedding_dim
        self.process_data()
    
    def process_data(self):
        """
        create nodes as number of unique genes
        get time column and time points to create a time series graph after
        first create graph with those time points then create the embeddings
        """
        unique_genes = pd.concat([self.df['Gene1_clean'], self.df['Gene2_clean']]).unique()
        self.node_map = {gene: idx for idx, gene in enumerate(unique_genes)}
        self.num_nodes = len(self.node_map)
        print(f"Number of nodes: {self.num_nodes}")
        
        self.time_cols = [col for col in self.df.columns if 'Time_' in col]
        self.time_points = sorted(list(set([float(col.split('_')[-1]) 
                                          for col in self.time_cols if 'Gene1' in col])))
        print(f"Found {len(self.time_points)} time points")
        
        G, self.edge_index, self.edge_attr = self.create_graph_structure(self.time_points[0])
        print(f"Graph structure created with {len(self.edge_attr)} edges")
        
        self.create_temporal_embeddings()
    
    def create_graph_structure(self, time_point):
        G = nx.Graph()
        
        genes = list(self.node_map.keys())
        G.add_nodes_from(genes)
        
        for _, row in self.df.iterrows():
            hic_weight = row['HiC_Interaction']
            compartment_sim = 1 if row['Gene1_Compartment'] == row['Gene2_Compartment'] else 0
            tad_dist = abs(row['Gene1_TAD_Boundary_Distance'] - row['Gene2_TAD_Boundary_Distance'])
            tad_sim = 1 / (1 + tad_dist)
            ins_sim = 1 / (1 + abs(row['Gene1_Insulation_Score'] - row['Gene2_Insulation_Score']))
            expr_sim = 1 / (1 + abs(row[f'Gene1_Time_{time_point}'] - row[f'Gene2_Time_{time_point}']))
            
            weight = (hic_weight * 0.4 + 
                     compartment_sim * 0.15 + 
                     tad_sim * 0.15 + 
                     ins_sim * 0.15 + 
                     expr_sim * 0.15)
            
            G.add_edge(row['Gene1_clean'], row['Gene2_clean'], weight=weight)
        
        edge_index = []
        edge_weights = []
        
        for u, v, d in G.edges(data=True):
            i, j = self.node_map[u], self.node_map[v]
            edge_index.extend([[i, j], [j, i]])
            edge_weights.extend([d['weight'], d['weight']])
        
        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        edge_attr = edge_weights.unsqueeze(1)
        
        return G, edge_index, edge_attr
    
    def create_temporal_embeddings(self):
        self.node_features = {}
        
        for t in self.time_points:
            print(f"Creating embeddings for time point {t}")
            
            G, _, _ = self.create_graph_structure(t)
            
            node2vec = Node2Vec(
                G,
                dimensions=self.embedding_dim,
                walk_length=10,
                num_walks=80,
                workers=1,
                p=1,
                q=1,
                weight_key='weight'
            )
            
            model = node2vec.fit(window=10, min_count=1)
   
            embeddings = []
            for gene in self.node_map.keys():
                emb = model.wv[gene]
                embeddings.append(emb)
            
            self.node_features[t] = torch.tensor(np.array(embeddings), dtype=torch.float)
        
        print(f"Created embeddings with dimension {self.embedding_dim}")
        print(f"Feature range: [{self.node_features[self.time_points[0]].min():.4f}, "
              f"{self.node_features[self.time_points[0]].max():.4f}]")
    
    def get_pyg_graph(self, time_point):
        return Data(
            x=self.node_features[time_point],
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            num_nodes=self.num_nodes
        )
    
    def get_temporal_sequences(self):
        sequences = []
        labels = []
        
        for i in range(len(self.time_points) - self.seq_len - self.pred_len + 1):
            seq_graphs = [self.get_pyg_graph(t) for t in self.time_points[i:i+self.seq_len]]
            label_graphs = [self.get_pyg_graph(t) for t in 
                          self.time_points[i+self.seq_len:i+self.seq_len+self.pred_len]]
            
            if i == 0:
                print("\nSequence information:")
                print(f"Input time points: {self.time_points[i:i+self.seq_len]}")
                print(f"Target time points: {self.time_points[i+self.seq_len:i+self.seq_len+self.pred_len]}")
                print(f"Feature dimension: {seq_graphs[0].x.shape[1]}")
            
            sequences.append(seq_graphs)
            labels.append(label_graphs)
        
        print(f"\nCreated {len(sequences)} sequences")
        return sequences, labels

def train_model(model, train_sequences, train_labels, val_sequences, val_labels, 
                num_epochs=80, learning_rate=0.0001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for seq, label in zip(train_sequences, train_labels):
            optimizer.zero_grad()
            
            output = model(seq)
            target = torch.stack([g.x for g in label]).mean(dim=0)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seq, label in zip(val_sequences, val_labels):
                output = model(seq)
                target = torch.stack([g.x for g in label]).mean(dim=0)
                val_loss += criterion(output, target).item()
        
        train_loss = total_loss/len(train_sequences)
        val_loss = val_loss/len(val_sequences)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}\n')
    
    return train_losses, val_losses

def train_model_with_early_stopping_combined_loss(
    model, train_sequences, train_labels, val_sequences, val_labels, 
    num_epochs=50, learning_rate=0.00007, patience=10, delta=1.0, threshold=1e-4):
   
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
            target = torch.stack([g.x for g in label]).mean(dim=0)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seq, label in zip(val_sequences, val_labels):
                output = model(seq)
                target = torch.stack([g.x for g in label]).mean(dim=0)
                val_loss += criterion(output, target).item()
        
        train_loss = total_loss / len(train_sequences)
        val_loss = val_loss / len(val_sequences)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}\n')
        
        # Early stopping
        if val_loss < best_val_loss - threshold:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'plottings_AttentionSTGCN_combined_loss_layer_5_data_split_corrected/best_model.pth') 
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    
    model.load_state_dict(torch.load('plottings_AttentionSTGCN_combined_loss_layer_5_data_split_corrected/best_model.pth'))
    return train_losses, val_losses

def analyze_gene_predictions(model, val_sequences, val_labels, dataset, save_dir='plottings_AttentionSTGCN_combined_loss_layer_5_data_split_corrected'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        all_predictions = []
        all_targets = []

        for seq,label in zip(val_sequences, val_labels):
            pred = model(seq)
            target = torch.stack([g.x for g in label]).mean(dim=0)
            all_predictions.append(pred.numpy())
            all_targets.append(target.numpy())

        predictions = np.stack(all_predictions)
        targets = np.stack(all_targets)

        genes = list(dataset.node_map.keys())

        gene_metrics = {}
        for i, gene in enumerate(genes):
            gene_pred = predictions[:, i, :]
            gene_target = targets[:, i, :]

            corr, _ = pearsonr(gene_pred.flatten(), gene_target.flatten())

            mse = np.mean((gene_pred - gene_target) ** 2)
            mae = np.mean(np.abs(gene_pred - gene_target))
            corr = np.corrcoef(gene_pred.flatten(), gene_target.flatten())[0, 1]

            gene_metrics[gene] = {
                'MSE': mse,
                'MAE': mae,
                'Correlation': corr
            }
        
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

            metrics_df = pd.DataFrame.from_dict(gene_metrics, orient='index')
            metrics_df.to_csv(os.path.join(save_dir, 'gene_metrics.csv'))

            top_genes = metrics_df.nlargest(5, 'Correlation')
            bottom_genes = metrics_df.nsmallest(5, 'Correlation')
            
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.bar(top_genes.index, top_genes['Correlation'])
            plt.title('Top 5 Best Predicted Genes')
            plt.xticks(rotation=45)
            
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

def analyze_interactions(model, val_sequences, val_labels, dataset, save_dir='plottings_AttentionSTGCN_combined_loss_layer_5_data_split_corrected'):
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        pred = model(val_sequences[0])
        target = torch.stack([g.x for g in val_labels[0]]).mean(dim=0)
        
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

def evaluate_model_performance(model, val_sequences, val_labels, dataset, save_dir='plottings_AttentionSTGCN_combined_loss_layer_5_data_split_corrected'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    metrics = {}
    
    with torch.no_grad():
        all_predictions = []
        all_targets = []
        
        for seq, label in zip(val_sequences, val_labels):
            pred = model(seq)
            target = torch.stack([g.x for g in label]).mean(dim=0)
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
            gene_target = targets[:, i, :]
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
def split_temporal_sequences(sequences, labels, train_size=0.8):
    """
    Split the temporal sequences into training and testing datasets while maintaining the sequential order.

    """
    split_index = int(len(sequences) * train_size)

    train_seq = sequences[:split_index]
    train_labels = labels[:split_index]
    test_seq = sequences[split_index:]
    test_labels = labels[split_index:]
    
    return train_seq, train_labels, test_seq, test_labels

if __name__ == "__main__":
    dataset = TemporalGraphDataset(
        csv_file='mapped/enhanced_interactions.csv',
        embedding_dim=64,
        seq_len=10,
        pred_len=1
    )
    
    sequences, labels = dataset.get_temporal_sequences()
    
    #train_seq, val_seq, train_labels, val_labels = train_test_split(
    #   sequences, labels, test_size=0.2, random_state=42
    #)
    train_seq, train_labels, val_seq, val_labels = split_temporal_sequences(sequences, labels, train_size=0.8)
    
    model = AttentionSTGCN(
    num_nodes=dataset.num_nodes,
    in_channels=64,
    hidden_channels=32,
    out_channels=64,
    num_layers=5
)
    
    #train_losses, val_losses = train_model(
    #   model, train_seq, train_labels, val_seq, val_labels
    #)

    train_losses, val_losses = train_model_with_early_stopping_combined_loss(
        model, train_seq, train_labels, val_seq, val_labels
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('plottings_AttentionSTGCN_combined_loss_layer_5_data_split_corrected/training_progress.png')

    print("\nAnalyzing predictions...")
    gene_metrics, avg_corr = analyze_gene_predictions(model, val_seq, val_labels, dataset)
    interaction_corr = analyze_interactions(model, val_seq, val_labels, dataset)
    
    #print("\nPrediction Summary:")
    #print(f"Average Gene Correlation: {np.mean([m['Correlation'] for m in gene_metrics.values()]):.4f}")
    print(f"Interaction Preservation: {interaction_corr:.4f}")

    metrics = evaluate_model_performance(model, val_seq, val_labels, dataset)
    print("\nModel Performance Summary:")
    print(f"Overall Pearson Correlation: {metrics['Overall']['Pearson_Correlation']:.4f}")
    print(f"RMSE: {metrics['Overall']['RMSE']:.4f}")
    print(f"R² Score: {metrics['Overall']['R2_Score']:.4f}")
    print(f"\nGene-wise Performance:")
    print(f"Mean Gene Correlation: {metrics['Gene_Performance']['Mean_Correlation']:.4f}")
    print(f"Best Performing Genes: {', '.join(metrics['Gene_Performance']['Best_Genes'])}")
    print(f"\nTemporal Stability:")
    print(f"Mean Temporal Correlation: {metrics['Temporal_Stability']['Mean_Temporal_Correlation']:.4f}")