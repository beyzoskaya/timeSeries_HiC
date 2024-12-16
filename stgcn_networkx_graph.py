import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation

class TemporalGraphDataset:
    def __init__(self, csv_file, seq_len=10, pred_len=1):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.df = pd.read_csv(csv_file)
        self.process_data()
        
    def process_data(self):
        self.time_cols = [col for col in self.df.columns if 'Time_' in col]
        self.time_points = sorted(list(set([float(col.split('_')[-1]) 
                                          for col in self.time_cols if 'Gene1' in col])))
        
        unique_genes = pd.concat([self.df['Gene1'], self.df['Gene2']]).unique()
        self.node_map = {gene: idx for idx, gene in enumerate(unique_genes)}
        self.num_nodes = len(self.node_map)
        print(f"Number of nodes: {self.num_nodes}")
        
        self.G = self.create_graph_structure()
        self.process_node_features()

    def create_adjacency_matrix(self):
        """Create adjacency matrix from HiC interactions"""
        adj_matrix = np.zeros((self.num_nodes, self.num_nodes))
        
        for _, row in self.df.iterrows():
            i = self.node_map[row['Gene1']]
            j = self.node_map[row['Gene2']]
            adj_matrix[i, j] = row['HiC_Interaction']
            adj_matrix[j, i] = row['HiC_Interaction']
        
        return adj_matrix

    def create_graph_structure(self):
        """Create graph structure from adjacency matrix"""
        # Create adjacency matrix
        adj_matrix = self.create_adjacency_matrix()
        
        # Create networkx graph
        G = nx.from_numpy_array(adj_matrix)
        
        # Convert to PyG format
        edge_index = []
        edge_weights = []
        
        for u, v, d in G.edges(data=True):
            edge_index.append([u, v])
            edge_weights.append(d['weight'])
        
        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        
        # Normalize edge weights
        edge_weights = (edge_weights - edge_weights.mean()) / edge_weights.std()
        
        self.edge_index = edge_index
        self.edge_attr = edge_weights.unsqueeze(1)
        
        print(f"Graph structure created:")
        print(f"Adjacency matrix shape: {adj_matrix.shape}")
        print(f"Edge index shape: {self.edge_index.shape}")
        print(f"Edge weights shape: {self.edge_attr.shape}")
        
        return G
    
    def process_node_features(self):
        """Process node features for each time point"""
        self.node_features = {}
        
        for t in self.time_points:
            features = []
            for gene in self.node_map.keys():
                gene_data = self.df[(self.df['Gene1'] == gene) | 
                                  (self.df['Gene2'] == gene)].iloc[0]
                
                # Get features
                expr = gene_data[f'Gene1_Time_{t}'] if gene == gene_data['Gene1'] \
                      else gene_data[f'Gene2_Time_{t}']
                comp = 1 if (gene == gene_data['Gene1'] and gene_data['Gene1_Compartment'] == 'A') or \
                          (gene == gene_data['Gene2'] and gene_data['Gene2_Compartment'] == 'A') else 0
                tad = gene_data['Gene1_TAD_Boundary_Distance'] if gene == gene_data['Gene1'] \
                      else gene_data['Gene2_TAD_Boundary_Distance']
                ins = gene_data['Gene1_Insulation_Score'] if gene == gene_data['Gene1'] \
                      else gene_data['Gene2_Insulation_Score']
                
                features.append([expr, comp, tad, ins])
            
            self.node_features[t] = torch.tensor(features, dtype=torch.float)
    
    def create_graph(self, time_point):
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
            # Create sequence of graphs
            seq_graphs = [self.create_graph(t) for t in self.time_points[i:i+self.seq_len]]
            
            # Create label graphs
            label_graphs = [self.create_graph(t) for t in 
                          self.time_points[i+self.seq_len:i+self.seq_len+self.pred_len]]
            
            sequences.append(seq_graphs)
            labels.append(label_graphs)
        
        return sequences, labels

class STGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(STGCNLayer, self).__init__()
        self.temporal_conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.spatial_conv = GCNConv(out_channels, out_channels)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        
    def forward(self, x, edge_index, edge_weight):
        batch_size = len(x)  # Number of graphs in sequence
        
        # Process temporal dimension
        x_combined = torch.stack(x)  # [seq_len, num_nodes, features]
        x_combined = x_combined.permute(1, 2, 0)  # [num_nodes, features, seq_len]
        x_combined = self.temporal_conv(x_combined)  # Temporal convolution
        x_combined = x_combined.permute(0, 2, 1)  # [num_nodes, seq_len, features]
        
        # Process spatial dimension for each time step
        output = []
        for t in range(x_combined.size(1)):
            x_t = x_combined[:, t, :]  # Get features for current time step
            # Apply GCN
            out_t = self.spatial_conv(x_t, edge_index, edge_weight)
            out_t = self.batch_norm(out_t)
            out_t = F.relu(out_t)
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
        """Process a sequence of graphs"""
        # Extract features, edge_index, and edge_weight from sequence
        x = [g.x for g in graph_sequence]  # List of node features for each time step
        edge_index = graph_sequence[0].edge_index  # Same for all time steps
        edge_weight = graph_sequence[0].edge_attr.squeeze()
        
        # Process through layers
        x = self.input_layer(x, edge_index, edge_weight)
        
        for layer in self.hidden_layers:
            x = layer(x, edge_index, edge_weight)
        
        # Apply output layer to each time step
        outputs = []
        for x_t in x:
            out_t = self.output_layer(x_t)
            outputs.append(out_t)
        
        # Combine outputs
        return torch.stack(outputs).mean(dim=0)  # Average across time steps

# Modify training loop
def train_model(model, train_sequences, train_labels, val_sequences, val_labels, 
                num_epochs=100, learning_rate=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for seq, label in zip(train_sequences, train_labels):
            optimizer.zero_grad()
            
            # Forward pass
            output = model(seq)
            target = torch.stack([g.x for g in label]).mean(dim=0)
            loss = criterion(output, target)
            
            # Backward pass
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

class GraphVisualizer:
    def __init__(self, dataset):
        self.dataset = dataset
        
    def plot_adjacency_matrix(self):
        """Visualize HiC interaction matrix"""
        adj_matrix = self.dataset.create_adjacency_matrix()
        
        plt.figure(figsize=(10, 10))
        # Create mappable object with imshow
        im = plt.imshow(adj_matrix, cmap='coolwarm')
        plt.colorbar(im, label='Interaction Strength')
        plt.title('HiC Interaction Matrix')
        plt.axis('off')
        plt.show()
        
    def plot_graph_structure(self):
        """Visualize graph structure"""
        plt.figure(figsize=(12, 12))
        
        # Get position layout
        pos = nx.spring_layout(self.dataset.G)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.dataset.G, pos, 
                             node_color='lightblue',
                             node_size=100)
        
        # Draw edges with weights as colors
        edges = self.dataset.G.edges()
        weights = [self.dataset.G[u][v]['weight'] for u, v in edges]
        
        nx.draw_networkx_edges(self.dataset.G, pos, 
                             edge_color=weights,
                             edge_cmap=plt.cm.coolwarm,
                             width=2, alpha=0.6)
        
        plt.title('Graph Structure (Node Connections)')
        plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.coolwarm), 
                    label='Edge Weight (HiC Interaction)')
        plt.show()
        
    def plot_temporal_patterns(self, num_nodes=5):
        """Visualize temporal expression patterns"""
        plt.figure(figsize=(15, 6))
        
        # Select random nodes
        selected_nodes = np.random.choice(self.dataset.num_nodes, num_nodes)
        
        # Plot expression over time
        for node in selected_nodes:
            expressions = [self.dataset.node_features[t][node, 0].item() 
                         for t in self.dataset.time_points]
            plt.plot(self.dataset.time_points, expressions, 
                    marker='o', label=f'Node {node}')
        
        plt.xlabel('Time Points')
        plt.ylabel('Expression Level')
        plt.title('Temporal Expression Patterns')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_feature_distributions(self):
        """Visualize node feature distributions"""
        # Get features for first time point
        features = self.dataset.node_features[self.dataset.time_points[0]].numpy()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        feature_names = ['Expression', 'Compartment', 'TAD Distance', 'Insulation Score']
        
        for i, (ax, name) in enumerate(zip(axes.flat, feature_names)):
            sns.histplot(features[:, i], ax=ax)
            ax.set_title(f'{name} Distribution')
            ax.set_xlabel(name)
        
        plt.tight_layout()
        plt.show()
        
    def plot_training_progress(self, train_losses, val_losses):
        """Visualize training progress"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def animate_temporal_graph(self):
        """Create animation of graph evolution"""
        fig, ax = plt.subplots(figsize=(10, 10))
        pos = nx.spring_layout(self.dataset.G)
        
        def update(frame):
            ax.clear()
            time_point = self.dataset.time_points[frame]
            features = self.dataset.node_features[time_point].numpy()
            
            # Draw nodes with expression as color
            nx.draw_networkx_nodes(self.dataset.G, pos,
                                 node_color=features[:, 0],
                                 cmap='viridis',
                                 node_size=100)
            
            # Draw edges
            nx.draw_networkx_edges(self.dataset.G, pos,
                                 alpha=0.2)
            
            ax.set_title(f'Time point: {time_point}')
            
        anim = FuncAnimation(fig, update, 
                           frames=len(self.dataset.time_points),
                           interval=500)
        plt.close()
        return anim

if __name__ == "__main__":

    dataset = TemporalGraphDataset('mapped/enhanced_interactions.csv', seq_len=10, pred_len=1)

    visualizer = GraphVisualizer(dataset)
    
    # Plot initial visualizations
    print("Plotting initial visualizations...")
    visualizer.plot_adjacency_matrix()
    visualizer.plot_graph_structure()
    visualizer.plot_temporal_patterns()
    visualizer.plot_feature_distributions()
    sequences, labels = dataset.get_temporal_sequences()
    
    # Print dataset information
    print("\nDataset information:")
    print(f"Number of sequences: {len(sequences)}")
    print(f"Sequence length: {len(sequences[0])}")
    
    # Print sample graph information
    sample_graph = sequences[0][0]
    print("\nSample graph structure:")
    print(f"Node features: {sample_graph.x.shape}")
    print(f"Edge index: {sample_graph.edge_index.shape}")
    print(f"Edge weights: {sample_graph.edge_attr.shape}")
    
    # Split data
    train_seq, val_seq, train_labels, val_labels = train_test_split(
        sequences, labels, test_size=0.2, random_state=42)
    
    # Create model
    model = STGCN(
        num_nodes=dataset.num_nodes,
        in_channels=4,  # expression, compartment, TAD, insulation
        hidden_channels=64,
        out_channels=4,
        num_layers=3
    )

    #train_model(model, train_seq, train_labels, val_seq, val_labels)

    train_losses, val_losses = train_model(model, train_seq, train_labels, 
                                         val_seq, val_labels)
    
    # Plot training progress
    visualizer.plot_training_progress(train_losses, val_losses)
    
    # Create temporal animation
    anim = visualizer.animate_temporal_graph()
    
    # Save animation
    anim.save('temporal_evolution.gif', writer='pillow')