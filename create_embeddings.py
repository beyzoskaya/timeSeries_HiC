import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
import torch
from sklearn.preprocessing import StandardScaler
import pickle
import os

class TemporalNode2VecWrapper:
    def __init__(self, dimensions=128, walk_length=80, num_walks=10, workers=4):
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers
        
    def create_graph_for_timepoint(self, df, time_point):
        """Create a graph for a specific time point"""
        G = nx.Graph()
        
        # Add all genes as nodes
        genes = pd.concat([df['Gene1'], df['Gene2']]).unique()
        G.add_nodes_from(genes)
        
        # Add edges with weights
        for _, row in df.iterrows():
            # Calculate edge weight combining all features
            hic_weight = row['HiC_Interaction']
            compartment_sim = 1 if row['Gene1_Compartment'] == row['Gene2_Compartment'] else 0
            tad_dist = abs(row['Gene1_TAD_Boundary_Distance'] - row['Gene2_TAD_Boundary_Distance'])
            tad_sim = 1 / (1 + tad_dist)
            ins_sim = 1 / (1 + abs(row['Gene1_Insulation_Score'] - row['Gene2_Insulation_Score']))
            expr_sim = 1 / (1 + abs(row[f'Gene1_Time_{time_point}'] - row[f'Gene2_Time_{time_point}']))
            
            # Combine weights
            weight = (hic_weight * 0.4 + 
                     compartment_sim * 0.15 + 
                     tad_sim * 0.15 + 
                     ins_sim * 0.15 + 
                     expr_sim * 0.15)
            
            G.add_edge(row['Gene1'], row['Gene2'], weight=weight)
        
        return G
    
    def get_embeddings_for_timepoint(self, graph):
        """Get node2vec embeddings for a graph"""
        # Initialize node2vec
        node2vec = Node2Vec(
            graph,
            dimensions=self.dimensions,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            workers=self.workers,
            p=1,  # Return parameter
            q=1,  # In-out parameter
            weight_key='weight'  # Use edge weights
        )
        
        # Train node2vec model
        model = node2vec.fit(window=10, min_count=1)
        
        # Get embeddings
        embeddings = {}
        for node in graph.nodes():
            embeddings[node] = model.wv[node]
            
        return embeddings
    
    def create_temporal_embeddings(self, df):
        """Create embeddings for all time points"""
        # Get time points
        time_cols = [col for col in df.columns if 'Time_' in col and 'Gene1' in col]
        time_points = sorted([float(col.split('_')[-1]) for col in time_cols])
        
        # Create embeddings for each time point
        temporal_embeddings = {}
        for t in time_points:
            print(f"Processing time point {t}")
            graph = self.create_graph_for_timepoint(df, t)
            temporal_embeddings[t] = self.get_embeddings_for_timepoint(graph)
        
        return temporal_embeddings

def create_and_save_features(df, output_dir='embeddings'):
    """Create and save temporal features"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize temporal node2vec
    temporal_n2v = TemporalNode2VecWrapper()
    
    # Create embeddings
    print("Creating temporal embeddings...")
    embeddings = temporal_n2v.create_temporal_embeddings(df)
    
    # Save embeddings
    with open(os.path.join(output_dir, 'temporal_embeddings.pkl'), 'wb') as f:
        pickle.dump(embeddings, f)
    
    # Create node mapping
    unique_genes = pd.concat([df['Gene1'], df['Gene2']]).unique()
    node_map = {gene: idx for idx, gene in enumerate(unique_genes)}
    
    with open(os.path.join(output_dir, 'node_map.pkl'), 'wb') as f:
        pickle.dump(node_map, f)
    
    return embeddings, node_map

def load_embeddings_for_stgcn(embedding_file, df):
    """Load embeddings and prepare for STGCN"""
    # Load embeddings
    with open(embedding_file, 'rb') as f:
        temporal_embeddings = pickle.load(f)
    
    # Create node features dictionary
    node_features = {}
    scaler = StandardScaler()
    
    # Get time points
    time_points = sorted(temporal_embeddings.keys())
    
    # Get unique genes
    unique_genes = pd.concat([df['Gene1'], df['Gene2']]).unique()
    
    for t in time_points:
        features = []
        for gene in unique_genes:
            # Get embedding
            embedding = temporal_embeddings[t][gene]
            
            # Get other features
            gene_data = df[(df['Gene1'] == gene) | (df['Gene2'] == gene)].iloc[0]
            
            # Additional features
            expr = gene_data[f'Gene1_Time_{t}'] if gene == gene_data['Gene1'] else gene_data[f'Gene2_Time_{t}']
            comp = 1 if ((gene == gene_data['Gene1'] and gene_data['Gene1_Compartment'] == 'A') or
                        (gene == gene_data['Gene2'] and gene_data['Gene2_Compartment'] == 'A')) else 0
            tad = gene_data['Gene1_TAD_Boundary_Distance'] if gene == gene_data['Gene1'] else gene_data['Gene2_TAD_Boundary_Distance']
            ins = gene_data['Gene1_Insulation_Score'] if gene == gene_data['Gene1'] else gene_data['Gene2_Insulation_Score']
            
            # Combine features
            combined_features = np.concatenate([
                embedding,
                [expr, comp, tad, ins]
            ])
            features.append(combined_features)
        
        # Convert to array and normalize
        features = np.array(features)
        if t == time_points[0]:  # Fit scaler on first time point
            scaler.fit(features)
        features_normalized = scaler.transform(features)
        
        # Convert to tensor
        node_features[t] = torch.tensor(features_normalized, dtype=torch.float)
    
    return node_features

# Usage example
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('mapped/enhanced_interactions.csv')
    
    # Create and save embeddings
    embeddings, node_map = create_and_save_features(df)
    
    # Load for STGCN
    node_features = load_embeddings_for_stgcn('embeddings/temporal_embeddings.pkl', df)
    
    # Print information
    print(f"\nNumber of time points: {len(node_features)}")
    print(f"Feature dimension: {node_features[list(node_features.keys())[0]].shape[1]}")