import os
import sys
import glob
import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F
from node2vec import Node2Vec
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv, TransformerConv
from scipy.stats import spearmanr  

data_dir = "data_1mb_resolution/"
output_dir = "outputs/"  

time_points = ['0.5hour', '1hour', '1.5hour', '2hour', '3hour', '4hour']
embedding_dim = 128
hidden_channels = 64
out_channels = 32
learning_rate = 0.01
epochs = 10

def load_kr_normalized_matrix(file_path):

    matrix = pd.read_csv(file_path, sep='\t', header=None).values
    print(f"Loaded matrix from {file_path}, shape: {matrix.shape}")
    return matrix

def prepare_chromosome_data(chrom):
    graph_sequence = []
    for tp in time_points:
        file_path = os.path.join(data_dir, tp, f"{chrom}_1mb.txt")
        
        if os.path.exists(file_path):
            matrix = load_kr_normalized_matrix(file_path)
            #graph = nx.from_numpy_matrix(matrix)  # Convert matrix to graph
            graph = nx.from_pandas_edgelist(pd.DataFrame(matrix, columns=['source', 'target', 'weight']), 
                                'source', 'target', edge_attr='weight')
            
            print(f"{tp} - {chrom} graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            
            embeddings = generate_node2vec_embeddings(graph, dimensions=embedding_dim)
            print(f"Generated Node2Vec embeddings, shape: {embeddings.shape}")
            
            # Convert to PyTorch Geometric Data
            pyg_graph = from_networkx(graph)
            pyg_graph.x = embeddings  
            graph_sequence.append(pyg_graph)
        else:
            print(f"Warning: File {file_path} does not exist. Skipping.")
    
    return graph_sequence

def generate_node2vec_embeddings(graph, dimensions=128, walk_length=30, num_walks=200, p=1, q=1):
    node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, p=p, q=q, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    embeddings = [model.wv[str(node)] for node in graph.nodes()]
    return torch.tensor(embeddings, dtype=torch.float)

class STGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=128): 
        super(STGCN, self).__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, out_channels) 
        self.temporal_att = TransformerConv(out_channels, out_channels, heads=1)  

    def forward(self, x, edge_index, edge_weight=None):
        print(f"Input x shape: {x.shape}")

        x = self.gcn1(x, edge_index, edge_weight)
        x = F.relu(x)
        print(f"After GCNConv 1 x shape: {x.shape}")

        x = self.gcn2(x, edge_index, edge_weight)
        print(f"After GCNConv 2 x shape: {x.shape}")

        x = self.temporal_att(x, edge_index)
        print(f"After TransformerConv x shape: {x.shape}")

        return x

def calculate_dSCC(prediction, target):
    pred_flat = prediction.detach().cpu().numpy().flatten()
    target_flat = target.detach().cpu().numpy().flatten()
    dSCC_value, _ = spearmanr(pred_flat, target_flat)
    return dSCC_value

def train_and_test_chromosome(chrom, graph_sequence, epochs=10):
    model = STGCN(embedding_dim, hidden_channels, out_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_dSCC = 0  

    chrom_output_dir = os.path.join(output_dir, chrom)
    os.makedirs(chrom_output_dir, exist_ok=True)


    for epoch in range(epochs):
        total_loss = 0
        for t in range(len(graph_sequence) - 1):  # Train on all but last time point
            pyg_graph = graph_sequence[t]
            x, edge_index, edge_weight = pyg_graph.x, pyg_graph.edge_index, pyg_graph.edge_attr
            optimizer.zero_grad()
            out = model(x, edge_index, edge_weight)
            loss = F.mse_loss(out, x)  
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} for {chrom}, Loss: {total_loss / len(graph_sequence)}")

     
        test_graph = graph_sequence[-1]
        x_test, edge_index_test, edge_weight_test = test_graph.x, test_graph.edge_index, test_graph.edge_attr
        out_test = model(x_test, edge_index_test, edge_weight_test)
        dSCC_value = calculate_dSCC(out_test, x_test)
        best_dSCC = max(best_dSCC, dSCC_value) 
        print(f"dSCC after Epoch {epoch + 1} for {chrom}: {dSCC_value}")

    model_save_path = os.path.join(chrom_output_dir, f"{chrom}_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model for {chrom} saved to {model_save_path}")

    dSCC_output_path = os.path.join(chrom_output_dir, f"{chrom}_best_dSCC.txt")
    with open(dSCC_output_path, 'w') as f:
        f.write(f"Best dSCC for {chrom}: {best_dSCC}\n")
    print(f"Best dSCC for {chrom} saved to {dSCC_output_path}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python train_chromosome.py <chromosome>")
        print("Example: python train_chromosome.py chr1")
        sys.exit(1)

    chrom = sys.argv[1]
    print(f"\nPreparing data for {chrom}...")
    graph_sequence = prepare_chromosome_data(chrom)
    print(f"Data preparation complete for {chrom}.")
    
    print(f"\nTraining model for {chrom}...")
    train_and_test_chromosome(chrom, graph_sequence, epochs=epochs)
    print(f"Training and testing complete for {chrom}.\n")

if __name__ == "__main__":
    main()
