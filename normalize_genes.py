from sklearn.preprocessing import normalize as sk_normalize
import numpy as np

def normalize_hic_weights(edge_weights, nodes):
    """
    Normalize HiC interaction weights using matrix normalization.
    
    Parameters:
    -----------
    edge_weights : list of tuples
        List of (gene1, gene2, weight) tuples
    nodes : list
        List of all gene names
    
    Returns:
    --------
    list of tuples
        Normalized (gene1, gene2, weight) tuples
    """
    # Create node index mapping
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    
    # Create matrix
    n = len(nodes)
    matrix = np.zeros((n, n))
    
    # Fill matrix with weights
    for gene1, gene2, weight in edge_weights:
        i, j = node_to_idx[gene1], node_to_idx[gene2]
        matrix[i, j] = weight
        matrix[j, i] = weight  # symmetric
    
    # Add small constant to avoid zeros
    matrix = matrix + 1e-10
    
    # Normalize matrix
    normalized_matrix = sk_normalize(matrix, norm='l1', axis=1)
    
    # Convert back to edge list
    normalized_edges = []
    for i in range(n):
        for j in range(i+1, n):  # upper triangle only to avoid duplicates
            if normalized_matrix[i, j] > 0:
                gene1, gene2 = nodes[i], nodes[j]
                normalized_edges.append((gene1, gene2, normalized_matrix[i, j]))
    
    return normalized_edges