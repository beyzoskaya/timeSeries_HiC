import torch
import torch.nn as nn

def process_batch(seq, label):
    """Process batch data for training."""
    # Input: Use full embeddings
    x = torch.stack([g.x for g in seq])  # [seq_len, num_nodes, features]
    x = x.permute(2, 0, 1).unsqueeze(0)  # [1, features, seq_len, nodes]
    
    # Target: Use only expression values
    target = torch.stack([g.x[:, -1] for g in label])  # [1, nodes] (expression values)
    target = target.unsqueeze(1).unsqueeze(0)  # [1, 1, 1, nodes]
    
    return x, target

def calculate_correlation(tensor):
    # tensor shape: [batch, channels, time, nodes]
    # Reshape to 2D for correlation
    tensor = tensor.squeeze(0) # remove batch
    tensor = tensor.view(tensor.size(0), -1) # [channels, time*nodes]
    return torch.corrcoef(tensor)