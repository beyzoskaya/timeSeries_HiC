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
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

def clean_gene_name(gene_name):
    """Clean gene name by removing descriptions and extra information"""
    if pd.isna(gene_name):
        return gene_name
    return gene_name.split('(')[0].strip()

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
    
    def forward(self, output, target):
        cos_sim = F.cosine_similarity(output, target, dim=1)
        return 1 - cos_sim.mean()

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
                workers=4,
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

class STGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(STGCNLayer, self).__init__()
        self.temporal_conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.spatial_conv = GCNConv(out_channels, out_channels)
        self.instance_norm = nn.InstanceNorm1d(out_channels, affine=True)
        
        nn.init.xavier_uniform_(self.temporal_conv.weight, gain=0.1)
        nn.init.constant_(self.temporal_conv.bias, 0.0)
    
    def forward(self, x, edge_index, edge_weight):
        x_stack = torch.stack(x)
        x_combined = x_stack.permute(1, 2, 0)  # [num_nodes, features, seq_len]
        
        x_combined = self.temporal_conv(x_combined)
        x_combined = torch.clamp(x_combined, min=-10, max=10)
        x_combined = self.instance_norm(x_combined)
        x_combined = F.relu(x_combined)
        
        x_combined = x_combined.permute(0, 2, 1)  # [num_nodes, seq_len, features]
        
        output = []
        for t in range(x_combined.size(1)):
            x_t = x_combined[:, t, :]
            edge_weight_norm = F.softmax(edge_weight, dim=0)
            out_t = self.spatial_conv(x_t, edge_index, edge_weight_norm)
            out_t = F.relu(out_t)
            out_t = torch.clamp(out_t, min=-10, max=10)
            output.append(out_t)
        
        return output

class STGCN(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels, num_layers=3):
        super(STGCN, self).__init__()
        
        self.num_layers = num_layers
        self.input_layer = STGCNLayer(in_channels, hidden_channels)
        
        self.hidden_layers = nn.ModuleList([
            STGCNLayer(hidden_channels, hidden_channels)
            for _ in range(num_layers-2)
        ])
        
        self.output_layer = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, graph_sequence):
        x = [g.x for g in graph_sequence]
        edge_index = graph_sequence[0].edge_index
        edge_weight = graph_sequence[0].edge_attr.squeeze()
        
        x = self.input_layer(x, edge_index, edge_weight)
        
        for layer in self.hidden_layers:
            x = layer(x, edge_index, edge_weight)
        
        outputs = []
        for x_t in x:
            out_t = self.output_layer(x_t)
            outputs.append(out_t)
        
        return torch.stack(outputs).mean(dim=0)

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

def train_model_with_early_stopping(model, train_sequences, train_labels, 
                                    val_sequences, val_labels, 
                                    num_epochs=80, learning_rate=0.0001, 
                                    patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
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
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth') 
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    
    model.load_state_dict(torch.load('best_model.pth'))
    return train_losses, val_losses

def train_model_with_cosine_similarity_and_early_stopping(
    model, train_sequences, train_labels, val_sequences, val_labels, 
    num_epochs=80, learning_rate=0.0001, threshold=1e-5, patience=10):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = CosineSimilarityLoss()
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
        
        # Early stopping based on threshold
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth') 
        elif abs(val_loss - best_val_loss) < threshold:
            print("Early stopping triggered due to minimal loss improvement.")
            break
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered due to lack of improvement.")
                break
    
    model.load_state_dict(torch.load('best_model.pth'))
    return train_losses, val_losses

def analyze_gene_predictions(model, val_sequences, val_labels, dataset, save_dir='plottings_cosine_sim'):
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
        
        return gene_metrics

def analyze_interactions(model, val_sequences, val_labels, dataset, save_dir='plottings_cosine_sim'):
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

def visualize_predictions(model, val_sequences, val_labels, save_dir='plottings_combined'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    with torch.no_grad():
        # Get predictions and targets for the first sequence
        seq = val_sequences[0]
        label = val_labels[0]
        
        pred = model(seq)
        target = torch.stack([g.x for g in label]).mean(dim=0)
   
        pred_np = pred.numpy()
        target_np = target.numpy()
        
        num_genes = min(6, pred_np.shape[0])
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i in range(num_genes):
            ax = axes[i]
            
            times = range(pred_np.shape[1]) 
            ax.plot(times, pred_np[i, :], 'r-', label='Predicted', alpha=0.7)
            ax.plot(times, target_np[i, :], 'b-', label='Actual', alpha=0.7)
            
            ax.set_title(f'Gene {i+1}')
            ax.set_xlabel('Embedding Dimension')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'time_series_predictions.png'))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.scatter(target_np.flatten(), pred_np.flatten(), alpha=0.1)
        plt.plot([target_np.min(), target_np.max()], 
                [target_np.min(), target_np.max()], 'r--', label='Perfect Prediction')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs Actual Values')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'prediction_scatter.png'))
        plt.close()
        
        errors = pred_np - target_np
        plt.figure(figsize=(10, 6))
        plt.hist(errors.flatten(), bins=50, alpha=0.7)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Errors')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'error_distribution.png'))
        plt.close()
        
        mse = np.mean((pred_np - target_np) ** 2)
        mae = np.mean(np.abs(pred_np - target_np))
        
        with open(os.path.join(save_dir, 'metrics.txt'), 'w') as f:
            f.write('Prediction Metrics:\n')
            f.write(f'MSE: {mse:.6f}\n')
            f.write(f'MAE: {mae:.6f}\n')


if __name__ == "__main__":
    dataset = TemporalGraphDataset(
        csv_file='mapped/enhanced_interactions.csv',
        embedding_dim=64,
        seq_len=10,
        pred_len=1
    )
    
    sequences, labels = dataset.get_temporal_sequences()
    
    train_seq, val_seq, train_labels, val_labels = train_test_split(
       sequences, labels, test_size=0.2, random_state=42
    )
    
    model = STGCN(
        num_nodes=dataset.num_nodes,
        in_channels=64,  # Same as embedding_dim
        hidden_channels=32,
        out_channels=64,  # Same as embedding_dim
        num_layers=3
    )
    
    #train_losses, val_losses = train_model(
    #   model, train_seq, train_labels, val_seq, val_labels
    #)

    train_losses, val_losses = train_model_with_cosine_similarity_and_early_stopping(
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
    plt.savefig('plottings_cosine_sim/training_progress.png')

    print("\nCreating prediction visualizations...")
    visualize_predictions(model, val_seq, val_labels)

    print("\nAnalyzing predictions...")
    gene_metrics = analyze_gene_predictions(model, val_seq, val_labels, dataset)
    interaction_corr = analyze_interactions(model, val_seq, val_labels, dataset)
    
    print("\nPrediction Summary:")
    print(f"Average Gene Correlation: {np.mean([m['Correlation'] for m in gene_metrics.values()]):.4f}")
    print(f"Interaction Preservation: {interaction_corr:.4f}")
