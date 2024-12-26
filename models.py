# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
#from torch_geometric_temporal.nn.attention import STGCN
from torch_geometric_temporal.nn.recurrent import DCRNN, TGCN, A3TGCN, AGCRN
from torch_geometric_temporal.nn.attention import ASTGCN, GMAN

class STGCNModel(nn.Module):
    """Spatio-Temporal Graph Convolutional Network"""
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels, kernel_size=3):
        super(STGCNModel, self).__init__()
        self.stgcn = STGCN(
            num_nodes=num_nodes,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            K=2  # Chebyshev filter size
        )
        self.linear = nn.Linear(hidden_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        
    def forward(self, graph_sequence):
        x = torch.stack([g.x for g in graph_sequence])  # [seq_len, num_nodes, features]
        edge_index = graph_sequence[0].edge_index
        edge_weight = graph_sequence[0].edge_attr.squeeze()
        
        # Process through STGCN
        out = self.stgcn(x, edge_index, edge_weight)
        out = self.linear(out)
        out = self.norm(out)
        
        return out

class TGCNModel(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels):
        super(TGCNModel, self).__init__()
        
        # Main TGCN layer
        self.tgcn = TGCN(
            in_channels=in_channels,    # Should match your input feature dimension
            out_channels=hidden_channels # Hidden dimension
        )
        
        # Output layer
        self.linear = nn.Linear(hidden_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        
    def forward(self, graph_sequence):
        x = graph_sequence[0].x
        print(f"Input feature shape: {x.shape}")  # Should be [num_nodes, in_channels]
    
        h = None  # Hidden state
        outputs = []
        
        for graph in graph_sequence:
            x = graph.x  # Shape: [num_nodes, in_channels]
            edge_index = graph.edge_index
            edge_weight = graph.edge_attr.squeeze()
            
            # Apply TGCN
            out, h = self.tgcn(x, edge_index, edge_weight, h)
            outputs.append(out)
        
        # Average temporal outputs
        out = torch.stack(outputs).mean(dim=0)
        print(f"TGCN output shape: {out.shape}")  # Should be [num_nodes, hidden_channels]
        
        # Final linear layer
        out = self.linear(out)
        out = self.norm(out)
        print(f"Final output shape: {out.shape}")  # Should be [num_nodes, out_channels]
        
        return out
    
class BaseSTGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BaseSTGCNLayer, self).__init__()
        self.temporal_conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.spatial_conv = GCNConv(out_channels, out_channels)
        self.instance_norm = nn.InstanceNorm1d(out_channels, affine=True)
        
        nn.init.xavier_uniform_(self.temporal_conv.weight, gain=0.1)
        nn.init.constant_(self.temporal_conv.bias, 0.0)
    
    def forward(self, x, edge_index, edge_weight):
        x_stack = torch.stack(x)
        x_combined = x_stack.permute(1, 2, 0)
        
        x_combined = self.temporal_conv(x_combined)
        x_combined = torch.clamp(x_combined, min=-10, max=10)
        x_combined = self.instance_norm(x_combined)
        x_combined = F.relu(x_combined)
        
        x_combined = x_combined.permute(0, 2, 1)
        
        output = []
        for t in range(x_combined.size(1)):
            x_t = x_combined[:, t, :]
            edge_weight_norm = F.softmax(edge_weight, dim=0)
            out_t = self.spatial_conv(x_t, edge_index, edge_weight_norm)
            out_t = F.relu(out_t)
            out_t = torch.clamp(out_t, min=-10, max=10)
            output.append(out_t)
        
        return output

class BaseSTGCN(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels, num_layers=3):
        super(BaseSTGCN, self).__init__()
        
        self.num_layers = num_layers
        self.input_layer = BaseSTGCNLayer(in_channels, hidden_channels)
        
        self.hidden_layers = nn.ModuleList([
            BaseSTGCNLayer(hidden_channels, hidden_channels)
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

class EnhancedSTGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EnhancedSTGCNLayer, self).__init__()
        self.temporal_conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.spatial_conv = GCNConv(out_channels, out_channels)
        self.layer_norm = nn.LayerNorm(out_channels)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        nn.init.xavier_uniform_(self.temporal_conv.weight, gain=0.1)
        nn.init.constant_(self.temporal_conv.bias, 0.0)
    
    def forward(self, x, edge_index, edge_weight):
        x_stack = torch.stack(x)
        identity = x_stack
        
        x_combined = x_stack.permute(1, 2, 0)
        x_combined = self.temporal_conv(x_combined)
        x_combined = self.layer_norm(x_combined.transpose(1, 2)).transpose(1, 2)
        x_combined = F.relu(x_combined)
        
        res = self.residual(identity.permute(1, 2, 0)).permute(2, 0, 1)
        x_combined = x_combined + res.permute(1, 2, 0)
        
        x_combined = x_combined.permute(0, 2, 1)
        
        output = []
        for t in range(x_combined.size(1)):
            x_t = x_combined[:, t, :]
            edge_weight_norm = F.softmax(edge_weight, dim=0)
            out_t = self.spatial_conv(x_t, edge_index, edge_weight_norm)
            out_t = F.relu(out_t)
            output.append(out_t)
        
        return output

class EnhancedSTGCN(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels, num_layers=3):
        super(EnhancedSTGCN, self).__init__()
        
        self.input_layer = EnhancedSTGCNLayer(in_channels, hidden_channels)
        
        self.path1 = nn.ModuleList([
            EnhancedSTGCNLayer(hidden_channels, hidden_channels)
            for _ in range(num_layers-2)
        ])
        
        self.path2 = nn.ModuleList([
            EnhancedSTGCNLayer(hidden_channels, hidden_channels//2)
            for _ in range(num_layers-2)
        ])
        
        self.combine = nn.Linear(hidden_channels + hidden_channels//2, hidden_channels)
        self.output_layer = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, graph_sequence):
        x = [g.x for g in graph_sequence]
        edge_index = graph_sequence[0].edge_index
        edge_weight = graph_sequence[0].edge_attr.squeeze()
        
        x = self.input_layer(x, edge_index, edge_weight)
        
        x1 = x
        x2 = x
        
        for layer1, layer2 in zip(self.path1, self.path2):
            x1 = layer1(x1, edge_index, edge_weight)
            x2 = layer2(x2, edge_index, edge_weight)
        
        outputs = []
        for t in range(len(x1)):
            combined = torch.cat([x1[t], x2[t]], dim=-1)
            combined = self.combine(combined)
            out_t = self.output_layer(combined)
            outputs.append(out_t)
        
        return torch.stack(outputs).mean(dim=0)

class AttentionSTGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionSTGCNLayer, self).__init__()
        self.temporal_conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.spatial_conv = GCNConv(out_channels, out_channels)
        self.layer_norm = nn.LayerNorm(out_channels)

        self.query = nn.Linear(out_channels, out_channels)
        self.key = nn.Linear(out_channels, out_channels)
        self.value = nn.Linear(out_channels, out_channels)

        nn.init.xavier_uniform_(self.temporal_conv.weight, gain=0.1)
        nn.init.constant_(self.temporal_conv.bias, 0.0)
    
    def forward(self, x, edge_index, edge_weight):
        x_stack = torch.stack(x)
        x_combined = x_stack.permute(1,2,0) # [num_nodes, features, seq_len]

        x_combined = self.temporal_conv(x_combined)
        x_combined = self.layer_norm(x_combined.transpose(1, 2)).transpose(1, 2)

        x_att = x_combined.permute(0,2,1)
        Q = self.query(x_att)
        K = self.key(x_att)
        V = self.value(x_att)

        # Scaled dot-product attention
        attention_weights = F.softmax(torch.bmm(Q, K.transpose(-2, -1)) / np.sqrt(x_att.size(-1)), dim=-1)
        x_att = torch.bmm(attention_weights, V)

        # Back to original format
        x_combined = x_att.permute(0, 2, 1)  # [num_nodes, features, seq_len]
        x_combined = F.relu(x_combined)

        output = []
        for t in range(x_combined.size(2)):
            x_t = x_combined[:, :, t]
            edge_weight_norm = F.softmax(edge_weight, dim=0)
            out_t = self.spatial_conv(x_t, edge_index, edge_weight_norm)
            out_t = F.relu(out_t)
            output.append(out_t)
        
        return output

class AttentionSTGCN(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels, num_layers=3):
        super(AttentionSTGCN, self).__init__()
        
        self.num_layers = num_layers
        self.input_layer = AttentionSTGCNLayer(in_channels, hidden_channels)
        
        self.hidden_layers = nn.ModuleList([
            AttentionSTGCNLayer(hidden_channels, hidden_channels)
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



