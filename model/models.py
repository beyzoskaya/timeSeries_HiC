import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from STGCN.model import layers

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
            self.relu = nn.ReLU()
            self.elu = nn.ELU()
            self.dropout = nn.Dropout(p=args.droprate)
        
        #FIXED projection layer is added for prediction of expression values
        self.expression_proj = nn.Sequential(
        nn.Linear(blocks[-1][0], 32),  # Keep more features initially
        nn.ELU(),
        nn.Dropout(p=0.1),
        nn.Linear(32, 16),
        nn.ELU(),
        nn.Linear(16, 1)
        )

        self.expression_proj_embedding_dim_16 = nn.Sequential(
        nn.Linear(blocks[-1][0], 16),  
        nn.ELU(),
        nn.Dropout(p=0.1),
        nn.Linear(16, 8),              
        nn.ELU(),
        nn.Linear(8, 1)               
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
        x = self.expression_proj(x)  # [batch, time_steps, nodes, 1]
        #print("After Projection Shape:", x.shape)
        x = x.permute(0, 3, 1, 2)  # [batch, 1, time_steps, nodes]
        
        return x 

class STGCNChebGraphConvProjectedGeneConnectedAttention(nn.Module):
    def __init__(self, args, blocks, n_vertex, gene_connections):
        super(STGCNChebGraphConvProjectedGeneConnectedAttention, self).__init__()
        
        connections = torch.tensor([gene_connections.get(i, 0) for i in range(n_vertex)])
        self.connection_weights = F.softmax(connections, dim=0)
        
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.STConvBlockLSTM(args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], 
                                            args.act_func, args.graph_conv_type, args.gso, 
                                            args.enable_bias, args.droprate))
        self.st_blocks = nn.Sequential(*modules)
        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko

        self.connectivity_attention = nn.Sequential(
            nn.Linear(blocks[-1][0], blocks[-1][0] // 2),
            nn.LayerNorm(blocks[-1][0] // 2),
            nn.ELU(),
            nn.Linear(blocks[-1][0] // 2, 1),
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
            self.elu = nn.ELU()
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
      
        connectivity_weights = self.connection_weights.to(x.device)
        connectivity_weights = connectivity_weights.view(1, 1, -1, 1)
        
        attention = (learned_attention * (1.0 + self.attention_scale * connectivity_weights))
        attention = F.layer_norm(attention, [attention.size(-1)])

        x = x * attention + 0.1 * x  # Small residual to maintain variation
        
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

