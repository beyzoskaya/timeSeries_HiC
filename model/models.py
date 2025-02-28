import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from STGCN.model import layers
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class STGCNChebGraphConv(nn.Module):
    # STGCNChebGraphConv contains 'TGTND TGTND TNFF' structure
    # ChebGraphConv is the graph convolution from ChebyNet.
    # Using the Chebyshev polynomials of the first kind as a graph filter.
        
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normalization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer

    def __init__(self, args, blocks, n_vertex):
        super(STGCNChebGraphConv, self).__init__()
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.STConvBlock(args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], args.act_func, args.graph_conv_type, args.gso, args.enable_bias, args.droprate))
        self.st_blocks = nn.Sequential(*modules)
        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko
        if self.Ko > 1:
            self.output = layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex, args.act_func, args.enable_bias, args.droprate)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], bias=args.enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], bias=args.enable_bias)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=args.droprate)

    def forward(self, x):
        x = self.st_blocks(x)
        if self.Ko > 1:
            x = self.output(x)
        
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)
            x = x * 1.5
        
        return x
    
class ScaledSTGCNChebGraphConv(STGCNChebGraphConv):
    def __init__(self, args, blocks, n_vertex):
        super(ScaledSTGCNChebGraphConv, self).__init__(args, blocks, n_vertex)
        # Add more aggressive scaling parameters
        self.scale_factor = nn.Parameter(torch.ones(1) * 3.0)  # Start with larger scale
        self.shift = nn.Parameter(torch.zeros(1))  # Allow for shift
        
    def forward(self, x):
        # Get base predictions
        x = super().forward(x)
        
        # Get the previous time step values
        prev_values = x[:, :, -2:-1, :] if x.size(2) > 1 else 0
        
        # Calculate changes from previous step
        changes = x[:, :, -1:, :] - prev_values
        
        # Scale the changes more aggressively
        scaled_changes = changes * self.scale_factor + self.shift
        
        # Apply scaled changes to previous values
        x = prev_values + scaled_changes
        
        return x

class STGCNChebGraphConvProjected(nn.Module):
    def __init__(self, args, blocks, n_vertex):
        super(STGCNChebGraphConvProjected, self).__init__()
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.STConvBlockLSTM(args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], 
                                            args.act_func, args.graph_conv_type, args.gso, 
                                            args.enable_bias, args.droprate))
        self.st_blocks = nn.Sequential(*modules)
        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko

        if self.Ko > 1:
            self.output = layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], 
                                           n_vertex, args.act_func, args.enable_bias, args.droprate)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], 
                                bias=args.enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], 
                                bias=args.enable_bias)
            #self.relu = nn.ReLU()
            self.elu = nn.ELU()
            self.dropout = nn.Dropout(p=args.droprate)
        
        #FIXED projection layer is added for prediction of expression values
        self.expression_proj = nn.Sequential(
        nn.Linear(blocks[-1][0], 32), 
        nn.ELU(),
        nn.Dropout(p=0.1),
        nn.Linear(32, 16),
        nn.ELU(),
        nn.Linear(16, 1)
        )

        self.expression_proj_miRNA = nn.Sequential(
            nn.Linear(blocks[-1][0], 64),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(64,32),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(32,16),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(16,1)
        )
        
    def forward(self, x):
        #print("Input Shape:", x.shape)
        # Original STGCN forward pass
        x = self.st_blocks(x)
        #print("After STConvBlockLSTM Shape:", x.shape)
        
        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.elu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)

        #print("After OutputBlock Shape:", x.shape)

        # Project to expression values
        batch_size, features, time_steps, nodes = x.shape
        x = x.permute(0, 2, 3, 1)  # [batch, time_steps, nodes, features]
        x = self.expression_proj_miRNA(x)  # [batch, time_steps, nodes, 1]
        #print("After Projection Shape:", x.shape)
        x = x.permute(0, 3, 1, 2)  # [batch, 1, time_steps, nodes]
        
        return x 
    
class STGCNChebGraphConvProjectedGeneConnectedAttention(nn.Module):
    def __init__(self, args, blocks, n_vertex, gene_connections):
        super(STGCNChebGraphConvProjectedGeneConnectedAttention, self).__init__()
        
        connections = torch.tensor([gene_connections.get(i, 0) for i in range(n_vertex)], 
                                 dtype=torch.float32)  # not implemented for Long error
        self.connection_weights = F.softmax(connections, dim=0)
        
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.STConvBlock(args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], 
                                            args.act_func, args.graph_conv_type, args.gso, 
                                            args.enable_bias, args.droprate))
        self.st_blocks = nn.Sequential(*modules)
        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko

        self.connectivity_attention = nn.Sequential(
            nn.Linear(blocks[-1][0], blocks[-1][0]//2),
            nn.LayerNorm(blocks[-1][0]//2 ),
            nn.ELU(),
            nn.Linear(blocks[-1][0]//2, 1),
            nn.Sigmoid()
        )
        
        self.attention_scale = nn.Parameter(torch.tensor(0.1))

        if self.Ko > 1:
            self.output = layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], 
                                           n_vertex, args.act_func, args.enable_bias, args.droprate)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], 
                                bias=args.enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], 
                                bias=args.enable_bias)
            self.relu = nn.ReLU()
            #self.elu = nn.ELU()
            self.dropout = nn.Dropout(p=args.droprate)
        
        self.expression_proj = nn.Sequential(
            nn.Linear(blocks[-1][0], 32),
            nn.LayerNorm(32),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.ELU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        identity = x
        
        x = self.st_blocks(x)
        
        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.elu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)

        batch_size, features, time_steps, nodes = x.shape
        x = x.permute(0, 2, 3, 1)  # [batch, time_steps, nodes, features]
        
        learned_attention = self.connectivity_attention(x)  # [batch, time_steps, nodes, 1]
      
        connectivity_weights = self.connection_weights
        connectivity_weights = connectivity_weights.view(1, 1, -1, 1)
        
        attention = (learned_attention * (1.0 + self.attention_scale * connectivity_weights))
        attention = F.layer_norm(attention, [attention.size(-1)])

        x = x * attention + 0.1 * x  # Small residual to maintain variation
        
        x = self.expression_proj(x)
        x = x.permute(0, 3, 1, 2)
        
        return x

# This model works best for mRNA predictions with enhanced_temporal_loss
class STGCNChebGraphConvProjectedGeneConnectedAttentionLSTM(nn.Module):
    def __init__(self, args, blocks, n_vertex, gene_connections):
        super(STGCNChebGraphConvProjectedGeneConnectedAttentionLSTM, self).__init__()
        
        connections = torch.tensor([gene_connections.get(i, 0) for i in range(n_vertex)], 
                                 dtype=torch.float32)
        self.connection_weights = F.softmax(connections, dim=0)
        
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.STConvBlock(args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], 
                                            args.act_func, args.graph_conv_type, args.gso, 
                                            args.enable_bias, args.droprate))
        self.st_blocks = nn.Sequential(*modules)
        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko

        # bidirectional LSTM 
        self.lstm = nn.LSTM(
            input_size=blocks[-3][-1],  # Input size is the feature dimension
            hidden_size=blocks[-3][-1],
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

        print(f"Hidden size blocks [-3][-1]: {blocks[-3][-1]}")
    
        self.lstm_proj = nn.Linear(2 * blocks[-3][-1], blocks[-3][-1])  # *2 for bidirectional
        
        self.lstm_norm = nn.LayerNorm([n_vertex, blocks[-3][-1]])
        
        self.connectivity_attention = nn.Sequential(
            nn.Linear(blocks[-1][0], blocks[-1][0]//2),
            nn.LayerNorm(blocks[-1][0]//2),
            nn.ELU(),
            nn.Linear(blocks[-1][0]//2, 1),
            nn.Sigmoid()
        )
        
        self.attention_scale = nn.Parameter(torch.tensor(0.1))

        if self.Ko > 1:
            self.output = layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], 
                                           n_vertex, args.act_func, args.enable_bias, args.droprate)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], 
                                bias=args.enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], 
                                bias=args.enable_bias)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=args.droprate)
        
        self.expression_proj = nn.Sequential(
            nn.Linear(blocks[-1][0], 32),
            nn.LayerNorm(32),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.ELU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        identity = x
        
        # ST-Blocks processing
        x = self.st_blocks(x)
        
        batch_size, features, time_steps, nodes = x.shape
        x_lstm = x.permute(0, 3, 2, 1)  # [batch, nodes, time_steps, features]
        x_lstm = x_lstm.reshape(batch_size * nodes, time_steps, features)
        
        lstm_out, _ = self.lstm(x_lstm)
        lstm_out = self.lstm_proj(lstm_out)
        lstm_out = lstm_out.reshape(batch_size, nodes, time_steps, features)
        lstm_out = lstm_out.permute(0, 3, 2, 1)  # Back to [batch, features, time_steps, nodes]
        
        # Residual connection with ST-Blocks output
        x = x + lstm_out
        x = self.lstm_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x).permute(0, 3, 1, 2)

        # Attention mechanism
        x = x.permute(0, 2, 3, 1)  # [batch, time_steps, nodes, features]
        learned_attention = self.connectivity_attention(x)
        
        connectivity_weights = self.connection_weights.view(1, 1, -1, 1)
        attention = (learned_attention * (1.0 + self.attention_scale * connectivity_weights))
        attention = F.layer_norm(attention, [attention.size(-1)])
        
        x = x * attention + 0.1 * x  # Residual connection
        
        # Final projection
        x = self.expression_proj(x)
        x = x.permute(0, 3, 1, 2)
        
        return x
    
class STGCNGraphConv(nn.Module):
    # STGCNGraphConv contains 'TGTND TGTND TNFF' structure
    # GraphConv is the graph convolution from GCN.
    # GraphConv is not the first-order ChebConv, because the renormalization trick is adopted.
    # Be careful about over-smoothing.
        
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (GraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (GraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normalization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer

    def __init__(self, args, blocks, n_vertex):
        super(STGCNGraphConv, self).__init__()
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.STConvBlock(args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], args.act_func, args.graph_conv_type, args.gso, args.enable_bias, args.droprate))
        self.st_blocks = nn.Sequential(*modules)
        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko
        if self.Ko > 1:
            self.output = layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex, args.act_func, args.enable_bias, args.droprate)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], bias=args.enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], bias=args.enable_bias)
            self.relu = nn.ReLU()
            self.do = nn.Dropout(p=args.droprate)

    def forward(self, x):
        x = self.st_blocks(x)
        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)
        
        return x


class STGCNGraphConvProjected(nn.Module):
   
    def __init__(self, args, blocks, n_vertex):
        super(STGCNGraphConvProjected, self).__init__()
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.STConvBlock(args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], args.act_func, args.graph_conv_type, args.gso, args.enable_bias, args.droprate))
        self.st_blocks = nn.Sequential(*modules)
        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko
        if self.Ko > 1:
            self.output = layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex, args.act_func, args.enable_bias, args.droprate)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], bias=args.enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], bias=args.enable_bias)
            self.relu = nn.ReLU()
            self.do = nn.Dropout(p=args.droprate)
        
        self.expression_proj = nn.Sequential(
        nn.Linear(blocks[-1][0], 16),  # Wider first projection
        nn.ReLU(),
        nn.Linear(16, 8),            # Gradual reduction
        nn.ReLU(),
        nn.Linear(8, 1)              # Final projection
    )
        

    def forward(self, x):
        x = self.st_blocks(x)
        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)
        
        batch_size, features, time_steps, nodes = x.shape
        x = x.permute(0, 2, 3, 1)  # [batch, time_steps, nodes, features]
        x = self.expression_proj(x)  # [batch, time_steps, nodes, 1]
        x = x.permute(0, 3, 1, 2)  # [batch, 1, time_steps, nodes]
        
        return x

class EnhancedSTGCNChebGraphConvProjected(nn.Module):
    def __init__(self, args, blocks, n_vertex):
        super(EnhancedSTGCNChebGraphConvProjected, self).__init__()

        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.STAttentionBlock(args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], 
                                          args.act_func, args.graph_conv_type, args.gso, 
                                          args.enable_bias, args.droprate))
        
        self.st_blocks = nn.Sequential(*modules)

        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        print(f"Ko: {Ko}")
        self.Ko = Ko

        if self.Ko > 1:
            self.output = layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], 
                                           n_vertex, args.act_func, args.enable_bias, args.droprate)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], 
                                bias=args.enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], 
                                bias=args.enable_bias)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=args.droprate)
        
        self.expression_proj = nn.Sequential(
        nn.Linear(blocks[-1][0], 16),  # Wider first projection
        nn.ReLU(),
        nn.Linear(16, 8),            # Gradual reduction
        nn.ReLU(),
        nn.Linear(8, 1)              # Final projection
    )
    
    def forward(self,x):

        x = self.st_blocks(x)

        if self.Ko > 1:
            #print(f"I am in Ko>1 part")
            x = self.output(x)
        elif self.Ko == 0:
            #print(f"I am in Ko==0 part")
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)

        batch_size, features, time_steps, nodes = x.shape
        x = x.permute(0, 2, 3, 1)  # [batch, time_steps, nodes, features]
        x = self.expression_proj(x)  # [batch, time_steps, nodes, 1]
        x = x.permute(0, 3, 1, 2)  # [batch, 1, time_steps, nodes]
        
        return x


class SmallSTGCN(nn.Module):
    def __init__(self, args, blocks, n_vertex):
        super(SmallSTGCN, self).__init__()

        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.SmallSTBlock(args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], 
                                          args.act_func, args.graph_conv_type, args.gso, 
                                          args.enable_bias, args.droprate))
        self.st_blocks = nn.Sequential(*modules)

        Ko = args.n_his - (len(blocks) - 3) * 2 * (min(args.Kt, 2) - 1)
        self.Ko = Ko
        
        if self.Ko > 1:
            self.output = layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], 
                                           n_vertex, args.act_func, args.enable_bias, args.droprate)
        else:
            self.fc1 = nn.Linear(blocks[-3][-1], blocks[-2][0], bias=args.enable_bias)
            self.fc2 = nn.Linear(blocks[-2][0], blocks[-1][0], bias=args.enable_bias)
            self.dropout = nn.Dropout(p=args.droprate)
        
        self.expression_proj = nn.Sequential(
        nn.Linear(blocks[-1][0], 8),  # Wider first projection
        nn.ReLU(),
        #nn.Linear(16, 8),            # Gradual reduction
        #nn.ReLU(),
        nn.Linear(8, 1)              # Final projection
    )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.st_blocks(x)
        
        if self.Ko > 1:
            x = self.output(x)
        else:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = F.elu(x)
            x = self.dropout(x)
            x = self.fc2(x).permute(0, 3, 1, 2)
        
        batch_size, features, time_steps, nodes = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.expression_proj(x)
        x = x.permute(0, 3, 1, 2)
        
        return x

class MiRNASTGCNWithAttention(nn.Module):
    def __init__(self, args, blocks, n_vertex):
        super(MiRNASTGCNWithAttention, self).__init__()
        
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.STConvBlock(
                args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], 
                args.act_func, args.graph_conv_type, args.gso, 
                args.enable_bias, args.droprate
            ))
        self.st_blocks = nn.Sequential(*modules)

        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko
        
        self.time_attention = nn.Sequential(
            nn.Linear(blocks[-3][-1], blocks[-3][-1] // 2),
            nn.LayerNorm(blocks[-3][-1] // 2),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(blocks[-3][-1] // 2, 1),
            nn.Sigmoid()
        )
        
        self.gene_attention = nn.Sequential(
            nn.Linear(blocks[-3][-1], blocks[-3][-1] // 2),
            nn.LayerNorm(blocks[-3][-1] // 2),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(blocks[-3][-1] // 2, 1),
            nn.Sigmoid()
        )
        
        self.temporal_context = nn.MultiheadAttention(
            embed_dim=blocks[-3][-1],
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm([n_vertex, blocks[-3][-1]])
        self.norm2 = nn.LayerNorm([n_vertex, blocks[-3][-1]])
        
        if self.Ko > 1:
            self.output = layers.OutputBlock(
                Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], 
                n_vertex, args.act_func, args.enable_bias, args.droprate
            )
        elif self.Ko == 0:
            self.fc1 = nn.Linear(
                in_features=blocks[-3][-1], 
                out_features=blocks[-2][0], 
                bias=args.enable_bias
            )
            self.fc2 = nn.Linear(
                in_features=blocks[-2][0], 
                out_features=blocks[-1][0], 
                bias=args.enable_bias
            )
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=args.droprate)
        
        self.expression_proj = nn.Sequential(
            nn.Linear(blocks[-1][0], 32),
            nn.LayerNorm(32),
            nn.ELU(),
            nn.Dropout(0.15),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.ELU(),
            nn.Dropout(0.15),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        x_st = self.st_blocks(x)
        
        batch_size, features, time_steps, nodes = x_st.shape
        x_reshaped = x_st.permute(0, 2, 3, 1)  # [batch, time_steps, nodes, features]
        
        x_temp = x_reshaped.reshape(batch_size * time_steps, nodes, features)
        x_temp, _ = self.temporal_context(x_temp, x_temp, x_temp)
        x_temp = x_temp.reshape(batch_size, time_steps, nodes, features)
   
        time_weights = self.time_attention(x_reshaped)  # [batch, time_steps, nodes, 1]
        x_time_weighted = x_reshaped * time_weights
        
        gene_weights = self.gene_attention(x_reshaped)  # [batch, time_steps, nodes, 1]
        x_combined = x_time_weighted * gene_weights + 0.1 * x_reshaped  # Residual connection
        
        x_norm = self.norm1(x_combined)
        
        x_norm = x_norm.permute(0, 3, 1, 2)  # [batch, features, time_steps, nodes]
        #print(f"Shape of x norm: {x_norm.shape}") [1,32,5,146]
    
        if self.Ko > 1:
            x_out = self.output(x_norm)
        elif self.Ko == 0:
            x_out = self.fc1(x_norm.permute(0, 2, 3, 1))
            x_out = self.relu(x_out)
            x_out = self.dropout(x_out)
            x_out = self.fc2(x_out).permute(0, 3, 1, 2)
    
        x_out = x_out.permute(0, 2, 3, 1)  # [batch, time_steps, nodes, features]
        x_proj = self.expression_proj(x_out)  # [batch, time_steps, nodes, 1]
        x_proj = x_proj.permute(0, 3, 1, 2)  # [batch, 1, time_steps, nodes]
        
        return x_proj


class MiRNASpecializedSTGCN(nn.Module):
    def __init__(self, args, blocks, n_vertex):
        super(MiRNASpecializedSTGCN, self).__init__()
        
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.STConvBlock(
                args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], 
                args.act_func, args.graph_conv_type, args.gso, 
                args.enable_bias, args.droprate
            ))
        self.st_blocks = nn.Sequential(*modules)
  
        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko
        
        self.spike_attention = nn.Sequential(
            nn.Linear(blocks[-3][-1], blocks[-3][-1]),
            nn.LayerNorm(blocks[-3][-1]),
            nn.GELU(),  
            nn.Dropout(0.2),
            nn.Linear(blocks[-3][-1], 1),
            nn.Sigmoid()
        )
        
        self.pattern_integration = nn.MultiheadAttention(
            embed_dim=blocks[-3][-1],
            num_heads=4,
            dropout=0.2,
            batch_first=True
        )
        
        if self.Ko > 1:
            self.output = layers.OutputBlock(
                Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], 
                n_vertex, args.act_func, args.enable_bias, args.droprate
            )
        elif self.Ko == 0:
            self.fc1 = nn.Linear(blocks[-3][-1], blocks[-2][0], bias=args.enable_bias)
            self.fc2 = nn.Linear(blocks[-2][0], blocks[-1][0], bias=args.enable_bias)
            self.act = nn.GELU()
            self.dropout = nn.Dropout(p=args.droprate)
        
        self.expression_proj = nn.Sequential(
            nn.Linear(blocks[-1][0], 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Linear(16, 8),
            nn.GELU(),
            nn.Linear(8, 1),
            nn.Softplus()  
        )

        self.skip_conn1 = nn.Linear(blocks[-3][-1], blocks[-1][0])
        self.norm1 = nn.LayerNorm(blocks[-1][0])
        
    def forward(self, x):
        x_st = self.st_blocks(x)
        
        batch_size, features, time_steps, nodes = x_st.shape
        x_reshaped = x_st.permute(0, 2, 3, 1)  # [batch, time_steps, nodes, features]
   
        x_skip = self.skip_conn1(x_reshaped)
        
        attention_weights = self.spike_attention(x_reshaped)
        x_attended = x_reshaped * attention_weights
        
        x_flat = x_attended.reshape(batch_size * time_steps, nodes, features)
        x_integrated, _ = self.pattern_integration(x_flat, x_flat, x_flat)
        x_integrated = x_integrated.reshape(batch_size, time_steps, nodes, features)
        
        if self.Ko > 1:
            x_integrated = x_integrated.permute(0, 3, 1, 2)  # [batch, features, time, nodes]
            x_out = self.output(x_integrated)
            x_out = x_out.permute(0, 2, 3, 1)  # [batch, time, nodes, features]
        elif self.Ko == 0:
            x_out = self.fc1(x_integrated)
            x_out = self.act(x_out)
            x_out = self.dropout(x_out)
            x_out = self.fc2(x_out)
        
        x_combined = x_out + x_skip
        x_combined = self.norm1(x_combined)
    
        x_final = self.expression_proj(x_combined)
        
        x_final = x_final.permute(0, 3, 1, 2)  # [batch, 1, time_steps, nodes]
        
        return x_final

class AdaptiveTemporalAttentionModel(nn.Module):
    def __init__(self, args, embedding_dim=32):
        super(AdaptiveTemporalAttentionModel, self).__init__()
        
        self.n_vertex = args.n_vertex  # Number of nodes --> number of genes
        self.n_his = args.n_his        # Historical sequence length --> how many time points are processed for the model
        self.n_pred = args.n_pred      # Prediction length --> one prediction at a time
        self.blocks = args.blocks      # Block structure --> block sizes 
        self.embedding_dim = embedding_dim 
        
        self.input_layer = nn.Linear(embedding_dim, self.blocks[0][0])
        
        self.temporal_layer = nn.Conv2d(
            in_channels=self.blocks[0][0],
            out_channels=self.blocks[0][1],
            kernel_size=(args.Kt, 1),
            padding=(1, 0)
        )
        
        self.query_layer = nn.Linear(self.blocks[0][1], self.blocks[0][1])
        self.key_layer = nn.Linear(self.blocks[0][1], self.blocks[0][1])
        self.value_layer = nn.Linear(self.blocks[0][1], self.blocks[0][1])
        self.scale = math.sqrt(self.blocks[0][1])
        
        self.output_layer = nn.Sequential(
            nn.Linear(self.blocks[0][1], self.blocks[1][0]),
            nn.ReLU(),
            nn.Dropout(args.droprate),
            nn.Linear(self.blocks[1][0], self.blocks[2][0]),
            nn.ReLU(),
            nn.Linear(self.blocks[2][0], args.n_pred)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        print(f"Input shape: {x.shape}")
        
        # Case 1: [batch, channels, time, nodes] (from STGCN data)
        if len(x.shape) == 4 and x.shape[1] < 5:  # Small number of channels
            # Convert to [batch, nodes, time, channels]
            x = x.permute(0, 3, 2, 1)
            print(f"Permuted from [B,C,T,N] to [B,N,T,C]: {x.shape}")
        
        # batch, nodes, time, features]
        nodes = x.shape[1]
        time_steps = x.shape[2]
        
        node_outputs = []
        
        for node_idx in range(nodes):
            node_features = x[:, node_idx, :, :]  # [batch, time, features]
            
            node_features = self.input_layer(node_features)  # [batch, time, block_dim]
            
            node_features = node_features.permute(0, 2, 1).unsqueeze(3)  # [batch, block_dim, time, 1]
            
            node_features = self.temporal_layer(node_features)  # [batch, block_dim, time-kt+1, 1]
            node_features = F.relu(node_features)
            
            node_features = node_features.squeeze(3).permute(0, 2, 1)  # [batch, time, block_dim]
            
            q = self.query_layer(node_features)
            k = self.key_layer(node_features)
            v = self.value_layer(node_features)
            
            attn_scores = torch.bmm(q, k.transpose(1, 2)) / self.scale
            attn_weights = F.softmax(attn_scores, dim=-1)
            context = torch.bmm(attn_weights, v)  # [batch, time, block_dim]
            
            pred_input = context[:, -1, :]  # [batch, block_dim]
        
            node_pred = self.output_layer(pred_input)  # [batch, n_pred]
            node_outputs.append(node_pred)
        
        # Stack predictions for all nodes: [batch, nodes, n_pred]
        outputs = torch.stack(node_outputs, dim=1)
        
        # [batch, 1, n_pred, nodes]
        outputs = outputs.permute(0, 2, 1).unsqueeze(1)
        
        return outputs