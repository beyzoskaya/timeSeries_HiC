# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
#from torch_geometric_temporal.nn.attention import STGCN
from torch_geometric_temporal.nn.recurrent import DCRNN, A3TGCN, TGCN, AGCRN
from torch_geometric.nn import GCNConv, ChebConv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN, TGCN
from torch_geometric.nn import GCNConv

class TemporalGNN(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels, 
                 window_size=5, model_type='TGCN'):
        super(TemporalGNN, self).__init__()
        
        self.num_nodes = num_nodes
        self.window_size = window_size
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.model_type = model_type
        
        # Temporal processing
        self.weight = nn.Parameter(torch.Tensor(window_size, in_channels, hidden_channels))
        self.bias = nn.Parameter(torch.Tensor(hidden_channels))
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.zeros_(self.bias)
        
        # Temporal attention
        self.attention_weight = nn.Parameter(torch.Tensor(hidden_channels, 1))
        torch.nn.init.xavier_uniform_(self.attention_weight)
        
        # GCN layers for spatial processing
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # Output projection
        self.linear = nn.Linear(hidden_channels, out_channels)
        
        # BatchNorm layers
        self.batch_norm1 = nn.BatchNorm1d(52)  # num_nodes
        self.batch_norm2 = nn.BatchNorm1d(52)  # num_nodes
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def temporal_attention(self, x):
        # x shape: [num_nodes, window_size, hidden_channels]
        attention_scores = torch.matmul(x, self.attention_weight)
        attention_weights = F.softmax(attention_scores, dim=1)
        attended_x = torch.sum(x * attention_weights, dim=1)
        return attended_x
        
    def forward(self, sequence):
        # Extract edge information
        edge_index = sequence[0].edge_index
        edge_weight = sequence[0].edge_attr.squeeze() if sequence[0].edge_attr is not None else None
        
        # Stack sequence [seq_len, num_nodes, features]
        x = torch.stack([graph.x for graph in sequence])
        print(f"Initial x shape: {x.shape}")
        
        # Reshape to [num_nodes, seq_len, features]
        x = x.permute(1, 0, 2).contiguous()
        print(f"After permute shape: {x.shape}")
        
        # Process each time step with learned weights
        hidden_states = []
        for t in range(self.window_size):
            h_t = torch.matmul(x[:, t, :], self.weight[t])
            hidden_states.append(h_t)
        
        # Stack hidden states
        h = torch.stack(hidden_states, dim=1)
        print(f"After temporal processing shape: {h.shape}")
        
        # Apply temporal attention
        h = self.temporal_attention(h)
        h = h + self.bias
        print(f"After attention shape: {h.shape}")  # [52, 64]
        
        # First GCN layer
        h1 = self.conv1(h, edge_index, edge_weight)
        h1 = h1.transpose(0, 1)  # [64, 52]
        h1 = self.batch_norm1(h1)
        h1 = h1.transpose(0, 1)  # [52, 64]
        h1 = self.dropout(torch.relu(h1))
        print(f"After first conv shape: {h1.shape}")
        
        # Second GCN layer
        h2 = self.conv2(h1, edge_index, edge_weight)
        h2 = h2.transpose(0, 1)  # [64, 52]
        h2 = self.batch_norm2(h2)
        h2 = h2.transpose(0, 1)  # [52, 64]
        h2 = self.dropout(torch.relu(h2))
        print(f"After second conv shape: {h2.shape}")
        
        # Output projection
        out = self.linear(h2)  # [52, 32]
        print(f"Final output shape: {out.shape}")
        
        return out
    
class TGCNModel(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels):
        super(TGCNModel, self).__init__()
        
        self.tgcn = TGCN(
            in_channels=in_channels,
            out_channels=hidden_channels
        )
        
        self.linear = nn.Linear(hidden_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        
    def forward(self, graph_sequence):
        outputs = []
        
        for graph in graph_sequence:
            x = graph.x
            edge_index = graph.edge_index
            edge_weight = graph.edge_attr.squeeze()
            
            # TGCN forward pass (returns only output)
            out = self.tgcn(x, edge_index, edge_weight)
            outputs.append(out)
        
        # Average temporal outputs
        out = torch.stack(outputs).mean(dim=0)
        
        # Final processing
        out = self.linear(out)
        #out = self.norm(out)
        
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



