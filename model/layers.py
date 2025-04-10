import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1))

    def forward(self, x):
        if self.c_in > self.c_out:
            x = self.align_conv(x)
        elif self.c_in < self.c_out:
            batch_size, _, timestep, n_vertex = x.shape
            x = torch.cat([x, torch.zeros([batch_size, self.c_out - self.c_in, timestep, n_vertex]).to(x)], dim=1)
        else:
            x = x
        
        return x

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, enable_padding=False, dilation=1, groups=1, bias=True):
        if enable_padding == True:
            self.__padding = (kernel_size - 1) * dilation
        else:
            self.__padding = 0
        super(CausalConv1d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=self.__padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[: , : , : -self.__padding]
        
        return result

class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, enable_padding=False, dilation=1, groups=1, bias=True):
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        dilation = nn.modules.utils._pair(dilation)
        if enable_padding == True:
            self.__padding = [int((kernel_size[i] - 1) * dilation[i]) for i in range(len(kernel_size))]
        else:
            self.__padding = 0
        self.left_padding = nn.modules.utils._pair(self.__padding)
        super(CausalConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, input):
        if self.__padding != 0:
            input = F.pad(input, (self.left_padding[1], 0, self.left_padding[0], 0))
        result = super(CausalConv2d, self).forward(input)

        return result

class TemporalConvLayer(nn.Module):
    def __init__(self, Kt, c_in, c_out, n_vertex, act_func):
        super(TemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.n_vertex = n_vertex
        self.align = Align(c_in, c_out)
        
        # Add padding to maintain temporal dimension
        if act_func == 'glu' or act_func == 'gtu':
            self.causal_conv = CausalConv2d(
                in_channels=c_in, 
                out_channels=2 * c_out, 
                kernel_size=(Kt, 1),
                enable_padding=True  # Enable padding
            )
        else:
            self.causal_conv = CausalConv2d(
                in_channels=c_in,
                out_channels=c_out,
                kernel_size=(Kt, 1),
                enable_padding=True  # Enable padding
            )
            
        self.relu = nn.ReLU()
        self.silu = nn.SiLU()
        self.act_func = act_func
        
    def forward(self, x):
        x_in = self.align(x)
        x_causal_conv = self.causal_conv(x)
        
        if self.act_func == 'glu':
            x_p = x_causal_conv[:, :self.c_out, :, :]
            x_q = x_causal_conv[:, -self.c_out:, :, :]
            x = torch.mul(x_p, torch.sigmoid(x_q))
        else:
            x = self.relu(x_causal_conv)
            
        return x

class ChebGraphConv(nn.Module):
    def __init__(self, c_in, c_out, Ks, gso, bias):
        super(ChebGraphConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.Ks = Ks
        self.gso = gso
        self.weight = nn.Parameter(torch.FloatTensor(Ks, c_in, c_out))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        #bs, c_in, ts, n_vertex = x.shape
        x = torch.permute(x, (0, 2, 3, 1))

        if self.Ks - 1 < 0:
            raise ValueError(f'ERROR: the graph convolution kernel size Ks has to be a positive integer, but received {self.Ks}.')  
        elif self.Ks - 1 == 0:
            x_0 = x
            x_list = [x_0]
        elif self.Ks - 1 == 1:
            x_0 = x
            x_1 = torch.einsum('hi,btij->bthj', self.gso, x)
            x_list = [x_0, x_1]
        elif self.Ks - 1 >= 2:
            x_0 = x
            x_1 = torch.einsum('hi,btij->bthj', self.gso, x)
            x_list = [x_0, x_1]
            for k in range(2, self.Ks):
                x_list.append(torch.einsum('hi,btij->bthj', 2 * self.gso, x_list[k - 1]) - x_list[k - 2])
        
        x = torch.stack(x_list, dim=2)

        cheb_graph_conv = torch.einsum('btkhi,kij->bthj', x, self.weight)

        if self.bias is not None:
            cheb_graph_conv = torch.add(cheb_graph_conv, self.bias)
        else:
            cheb_graph_conv = cheb_graph_conv
        
        return cheb_graph_conv

class GraphConv(nn.Module):
    def __init__(self, c_in, c_out, gso, bias):
        super(GraphConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.gso = gso
        self.weight = nn.Parameter(torch.FloatTensor(c_in, c_out))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        #bs, c_in, ts, n_vertex = x.shape
        x = torch.permute(x, (0, 2, 3, 1))

        first_mul = torch.einsum('hi,btij->bthj', self.gso, x)
        second_mul = torch.einsum('bthi,ij->bthj', first_mul, self.weight)

        if self.bias is not None:
            graph_conv = torch.add(second_mul, self.bias)
        else:
            graph_conv = second_mul
        
        return graph_conv

class GraphConvLayer(nn.Module):
    def __init__(self, graph_conv_type, c_in, c_out, Ks, gso, bias):
        super(GraphConvLayer, self).__init__()
        self.graph_conv_type = graph_conv_type
        self.c_in = c_in
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        self.Ks = Ks
        self.gso = gso
        if self.graph_conv_type == 'cheb_graph_conv':
            self.cheb_graph_conv = ChebGraphConv(c_out, c_out, Ks, gso, bias)
        elif self.graph_conv_type == 'graph_conv':
            self.graph_conv = GraphConv(c_out, c_out, gso, bias)

    def forward(self, x):
        x_gc_in = self.align(x)
        if self.graph_conv_type == 'cheb_graph_conv':
            x_gc = self.cheb_graph_conv(x_gc_in)
        elif self.graph_conv_type == 'graph_conv':
            x_gc = self.graph_conv(x_gc_in)
        x_gc = x_gc.permute(0, 3, 1, 2)
        x_gc_out = torch.add(x_gc, x_gc_in)

        return x_gc_out

class STConvBlock(nn.Module):
    # STConv Block contains 'TGTND' structure
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv or GraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    def __init__(self, Kt, Ks, n_vertex, last_block_channel, channels, act_func, graph_conv_type, gso, bias, droprate):
        super(STConvBlock, self).__init__()
        self.gso = gso
        self.tmp_conv1 = TemporalConvLayer(Kt, last_block_channel, channels[0], n_vertex, act_func)
        self.graph_conv = GraphConvLayer(graph_conv_type, channels[0], channels[1], Ks, gso, bias)
        self.tmp_conv2 = TemporalConvLayer(Kt, channels[1], channels[2], n_vertex, act_func)

        self.attention = nn.MultiheadAttention(
            embed_dim=channels[2],  # Feature dimension
            num_heads=4,            # Number of attention heads
            dropout=0.5,
            batch_first=True
        )
        self.tc2_ln = nn.LayerNorm([n_vertex, channels[2]], eps=1e-12)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x):
        #print(f"\nSTConvBlock input shape: {x.shape}")
        #if x.shape[2] < self.Kt:
        #    print(f"WARNING: Time dimension ({x.shape[2]}) is smaller than kernel size ({self.Kt})")
        x = self.tmp_conv1(x)
        #print(f"After first temporal conv: {x.shape}")
        x = self.graph_conv(x)
        #print(f"After graph conv: {x.shape}")
        x = self.elu(x)
        x = self.tmp_conv2(x)
        x = self.attention(x)
        #print(f"After second temporal conv: {x.shape}")
        x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        #print(f"After layer norm: {x.shape}")
        x = self.dropout(x)

        return x 

class FeatureFusionModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(channels*2, channels),
            nn.Sigmoid()
        )
    def forward(self, temporal_features, spatial_features):
        # temporal_feat: features from LSTM (temporal information)
        # spatial_feat: features from graph conv (spatial information)
        combined = torch.cat([temporal_features, spatial_features], dim=1)
        attention_weights = self.attention(combined)
        return temporal_features * attention_weights + spatial_features * (1- attention_weights)

class STConvBlockLSTM(nn.Module):
    def __init__(self, Kt, Ks, n_vertex, last_block_channel, channels, act_func, graph_conv_type, gso, bias, droprate):
        super(STConvBlockLSTM, self).__init__()
        self.gso = gso
        self.tmp_conv1 = TemporalConvLayer(Kt, last_block_channel, channels[0], n_vertex, act_func)
        self.graph_conv = GraphConvLayer(graph_conv_type, channels[0], channels[1], Ks, gso, bias)
        self.tmp_conv2 = TemporalConvLayer(Kt, channels[1], channels[2], n_vertex, act_func)

        self.lstm = nn.LSTM(
            input_size=channels[2],
            hidden_size=channels[2],
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=droprate 
        )
        
        self.tc2_ln = nn.LayerNorm([n_vertex, channels[2]], eps=1e-12)
        # When using more than 1 blocks for the STConv, we need to change the dimension as 2 * features because bidirectional LSTM doubled the feature size
        #self.tc2_ln = nn.LayerNorm([n_vertex, 2 * channels[2]], eps=1e-12)  # Update for bidirectional
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=droprate)
    
    def forward(self, x):
        #print("Input Shape:", x.shape)
        x = self.tmp_conv1(x)
        #print("After TemporalConv1 Shape:", x.shape)
        
        x = self.graph_conv(x)
        x = self.relu(x)
        #print("After GraphConv Shape:", x.shape)
        
        x = self.tmp_conv2(x)
        #print("After TemporalConv2 Shape:", x.shape)
        
        # Reshape for LSTM: (batch, seq_len, features) -> (batch, seq_len, channels[2])
        batch_size, features, seq_len, nodes = x.shape
        #print("Batch Size:", batch_size)  
        #print("Features:", features)  
        #print("Seq Len:", seq_len)  
        #print("Nodes:", nodes)

        x = x.permute(0, 2, 3, 1)  # [batch, seq_len, nodes, features]
        #print("After Permute Shape:", x.shape)
        x = x.reshape(batch_size * nodes, seq_len, features)  # [batch * nodes, seq_len, features]
        #print("After Reshape Shape:", x.shape)
        
        x, _ = self.lstm(x)  # Output shape: [batch * nodes, seq_len, channels[2]]
        #print("After LSTM Shape:", x.shape)
        
        # Reshape back to original shape
        x = x.reshape(batch_size, nodes, seq_len, features)
        # When using more than 1 blocks for the STConv, we need to change the dimension as 2 * features because bidirectional LSTM doubled the feature size
        #x = x.reshape(batch_size, nodes, seq_len, 2 * features) # bidirectional LSTM doubles the size 16 to 32
        #print("After Reshape Back Shape:", x.shape) 
        x = x.permute(0, 3, 2, 1)  # [batch, features, seq_len, nodes]
        #print("After Final Permute Shape:", x.shape)

        x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        #print("After LayerNorm Shape:", x.shape) 
    
        x = self.dropout(x)
        #print("After Dropout Shape:", x.shape)
        
        return x

class STConvBlockRNN(nn.Module):
    def __init__(self, Kt, Ks, n_vertex, last_block_channel, channels, act_func, graph_conv_type, gso, bias, droprate):
        super(STConvBlockRNN, self).__init__()
        self.gso = gso
        self.tmp_conv1 = TemporalConvLayer(Kt, last_block_channel, channels[0], n_vertex, act_func)
        self.graph_conv = GraphConvLayer(graph_conv_type, channels[0], channels[1], Ks, gso, bias)
        self.tmp_conv2 = TemporalConvLayer(Kt, channels[1], channels[2], n_vertex, act_func)

        self.rnn = nn.RNN(
            input_size=channels[2]//2,
            hidden_size=channels[2]//2,
            num_layers=2,    
            batch_first=True,
            dropout=0.1
        )

        self.tc2_ln = nn.LayerNorm([n_vertex, channels[2]], eps=1e-12)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=droprate)
    
    def forward(self, x):
        x = self.tmp_conv1(x)
        x = self.graph_conv(x)
        x = self.elu(x)
        x = self.tmp_conv2(x)
        
        batch_size, features, time_steps, nodes = x.shape
        x = x.permute(0, 2, 3, 1)  # [batch, time_steps, nodes, features]
        x = x.reshape(batch_size * nodes, time_steps, features)
        
        x, _ = self.rnn(x)
        
        x = x.reshape(batch_size, time_steps, nodes, features)
        x = x.permute(0, 3, 1, 2)  # [batch, features, time_steps, nodes]

        x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.dropout(x)
        
        return x
    
class STConvBlockLSTMFusionModule(nn.Module):
    def __init__(self, Kt, Ks, n_vertex, last_block_channel, channels, act_func, graph_conv_type, gso, bias, droprate):
        super(STConvBlockLSTMFusionModule, self).__init__()
        self.gso = gso

        self.tmp_conv1 = TemporalConvLayer(Kt, last_block_channel, channels[0], n_vertex, act_func)
        self.graph_conv = GraphConvLayer(graph_conv_type, channels[0], channels[1], Ks, gso, bias)
        self.tmp_conv2 = TemporalConvLayer(Kt, channels[1], channels[2], n_vertex, act_func)

        self.lstm = nn.LSTM(
            input_size=channels[2],
            hidden_size=channels[2],
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=droprate 
        )
        
        self.fusion = FeatureFusionModule(channels[2])
        
        self.spatial_proj = nn.Linear(channels[1], channels[2])
        
        self.tc2_ln = nn.LayerNorm([n_vertex, channels[2]], eps=1e-12)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=droprate)
    
    def forward(self, x):
        x = self.tmp_conv1(x)
        
        spatial_features = self.graph_conv(x)
        x = self.elu(spatial_features)
        
        spatial_skip = self.spatial_proj(spatial_features.permute(0, 2, 3, 1))
        
        x = self.tmp_conv2(x)
        
        batch_size, features, seq_len, nodes = x.shape
        x = x.permute(0, 2, 3, 1)  # [batch, seq_len, nodes, features]
        x = x.reshape(batch_size * nodes, seq_len, features)
        
        x, _ = self.lstm(x)
    
        x = x.reshape(batch_size, nodes, seq_len, features)
        x = self.fusion(x, spatial_skip)  # Fuse temporal and spatial
        
        x = x.permute(0, 3, 2, 1)  # [batch, features, seq_len, nodes]
        x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.dropout(x)
        
        return x

class STConvBlockGRU(nn.Module):
    def __init__(self, Kt, Ks, n_vertex, last_block_channel, channels, act_func, graph_conv_type, gso, bias, droprate):
        super(STConvBlockGRU, self).__init__()
        self.gso = gso
        self.tmp_conv1 = TemporalConvLayer(Kt, last_block_channel, channels[0], n_vertex, act_func)
        self.graph_conv = GraphConvLayer(graph_conv_type, channels[0], channels[1], Ks, gso, bias)
        self.tmp_conv2 = TemporalConvLayer(Kt, channels[1], channels[2], n_vertex, act_func)

        self.gru = nn.GRU(
            input_size=channels[2],  # Input feature size
            hidden_size=channels[2], # Hidden state size
            num_layers=2,            # Number of GRU layers
            batch_first=True         # Input shape: (batch, seq_len, features)
        )

        self.tc2_ln = nn.LayerNorm([n_vertex, channels[2]], eps=1e-12)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x):
        x = self.tmp_conv1(x)
        
        x = self.graph_conv(x)
        x = self.elu(x)
        
        x = self.tmp_conv2(x)
        
        # Reshape for GRU: (batch, seq_len, features) -> (batch, seq_len, channels[2])
        batch_size, features, seq_len, nodes = x.shape
        x = x.permute(0, 2, 3, 1)  # [batch, seq_len, nodes, features]
        x = x.reshape(batch_size * nodes, seq_len, features)  # [batch * nodes, seq_len, features]

        x, _ = self.gru(x)  # Output shape: [batch * nodes, seq_len, channels[2]]
        
        # Reshape back to original shape
        x = x.reshape(batch_size, nodes, seq_len, features)
        x = x.permute(0, 3, 2, 1)  # [batch, features, seq_len, nodes]

        x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        x = self.dropout(x)
        
        return x

class STConvBlockMultiHeadAttention(nn.Module):
    # STConv Block contains 'TGTND' structure with added attention
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv or GraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # A: Multi-head Attention
    # N: Layer Normolization
    # D: Dropout

    def __init__(self, Kt, Ks, n_vertex, last_block_channel, channels, act_func, graph_conv_type, gso, bias, droprate):
        super(STConvBlockMultiHeadAttention, self).__init__()
        self.gso = gso
        self.tmp_conv1 = TemporalConvLayer(Kt, last_block_channel, channels[0], n_vertex, act_func)
        self.graph_conv = GraphConvLayer(graph_conv_type, channels[0], channels[1], Ks, gso, bias)
        self.tmp_conv2 = TemporalConvLayer(Kt, channels[1], channels[2], n_vertex, act_func)
        
        # Add MultiheadAttention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=channels[2],  # Feature dimension
            num_heads=4,            # Number of attention heads
            dropout=droprate,
            batch_first=True
        )
        
        # Additional layer norm for attention
        self.attn_norm = nn.LayerNorm([n_vertex, channels[2]], eps=1e-12)
        
        self.tc2_ln = nn.LayerNorm([n_vertex, channels[2]], eps=1e-12)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x):
        # First temporal conv
        x = self.tmp_conv1(x)
        
        # Graph conv
        x = self.graph_conv(x)
        x = self.relu(x)
        
        # Second temporal conv
        x = self.tmp_conv2(x)
        
        # Prepare for attention
        batch_size, channels, time_steps, nodes = x.shape
        x = x.permute(0, 2, 3, 1)  # [batch, time, nodes, channels]
        
        # Reshape for attention
        x_reshaped = x.reshape(batch_size * nodes, time_steps, channels)
        
        # Apply attention
        attn_out, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        
        # Reshape back
        x = attn_out.reshape(batch_size, time_steps, nodes, channels)
        
        # Apply attention layer norm
        x = self.attn_norm(x)
        
        # Final layer norm and dropout
        x = x.permute(0, 3, 1, 2)  # [batch, channels, time, nodes]
        x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.dropout(x)

        return x

class OutputBlock(nn.Module):
    # Output block contains 'TNFF' structure
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer

    def __init__(self, Ko, last_block_channel, channels, end_channel, n_vertex, act_func, bias, droprate):
        super(OutputBlock, self).__init__()
        self.tmp_conv1 = TemporalConvLayer(Ko, last_block_channel, channels[0], n_vertex, act_func)
        self.fc1 = nn.Linear(in_features=channels[0], out_features=channels[1], bias=bias)
        self.fc2 = nn.Linear(in_features=channels[1], out_features=end_channel, bias=bias)
        self.tc1_ln = nn.LayerNorm([n_vertex, channels[0]], eps=1e-12)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        #self.dropout = nn.Dropout(p=droprate)

    def forward(self, x):
        x = self.tmp_conv1(x)
        x = self.tc1_ln(x.permute(0, 2, 3, 1))
        x = self.fc1(x)
        x = self.relu(x)
        #x = self.dropout(x)
        x = self.fc2(x).permute(0, 3, 1, 2)

        return x

class TemporalSelfAttention(nn.Module):
    def __init__(self, in_channels, n_vertex):
        super(TemporalSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.n_vertex = n_vertex

        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)

        self.out_proj = nn.Linear(in_channels, in_channels)
    
    def forward(self, x):
        batch_size, channels, time_steps, nodes = x.shape
        x = x.permute(0, 3, 2, 1) # [batch_size, nodes, time_steps, channels]

        # Q,K,V 
        q = self.query(x) # [batch_size, nodes, time_steps, channels]
        k = self.key(x) 
        v = self.value(x)

        # scale dot product 
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.in_channels).float())
        attention = F.softmax(scores, dim=-1)

        # attention
        out = torch.matmul(attention, v) # [batch_size, nodes, time_steps, channels]
        out = self.out_proj(out)

        # residual connection
        out = out + x

        out = out.permute(0,3,2,1) # [batch_size, nodes, time_steps, channels]
        return out 

class STAttentionBlock(nn.Module):
    def __init__(self, Kt, Ks, n_vertex, last_block_channel, channels, act_func, graph_conv_type, gso, bias, droprate):
        super(STAttentionBlock, self).__init__()

        self.tmp_conv1 = TemporalConvLayer(Kt, last_block_channel, channels[0], n_vertex, act_func)
        self.graph_conv = GraphConvLayer(graph_conv_type, channels[0], channels[1], Ks, gso, bias)
        self.tmp_conv2 = TemporalConvLayer(Kt, channels[1], channels[2], n_vertex, act_func)

        # temporal self-attention
        self.temporal_attention = TemporalSelfAttention(channels[2], n_vertex)

        self.tc2_ln = nn.LayerNorm([n_vertex, channels[2]], eps=1e-12)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)
    
    def forward(self, x):
        x = self.tmp_conv1(x)
        x = self.graph_conv(x)
        x = self.relu(x)
        x = self.tmp_conv2(x)

        x = self.temporal_attention(x)

        x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.dropout(x)
        return x

class TemporalGatingUnit(nn.Module):
    def __init__(self, channels):
        super(TemporalGatingUnit, self).__init__()
        self.channels = channels
        self.gate = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        gates = self.gate(x)
        return x * gates

class SmallTemporalAttention(nn.Module):
    def __init__(self, in_channels, n_vertex):
        super(SmallTemporalAttention, self).__init__()
        self.in_channels = in_channels
        self.n_vertex = n_vertex

        # Reduce parameter count in attention
        hidden_dim = max(in_channels // 2, 16)
        self.query = nn.Linear(in_channels, hidden_dim)
        self.key = nn.Linear(in_channels, hidden_dim)
        self.value = nn.Linear(in_channels, hidden_dim)

        self.gating = TemporalGatingUnit(hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, in_channels)
        self.layer_norm = nn.LayerNorm(in_channels)
    
    def forward(self, x):
        batch_size, channels, time_steps, nodes = x.shape
        x = x.permute(0, 3, 2, 1) # [batch_size, nodes, time_steps, channels]

        residual = x

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # scale dot product with temp scaling
        temperature = torch.sqrt(torch.tensor(q.size(-1))).to(x.device)
        scores = torch.matmul(q, k.transpose(-2, -1)) / temperature

        position_bias = self._get_relative_positions(time_steps)
        scores = scores + position_bias

        attention = F.softmax(scores, dim=-1)

        out = torch.matmul(attention, v)
        out = self.gating(out)
        
        out = self.out_proj(out)
        out = self.layer_norm(out + residual)
        
        return out.permute(0, 3, 2, 1)

    def _get_relative_positions(self, length):
        positions = torch.arange(length).float()
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        return torch.exp(-torch.abs(relative_positions) / length)
    
class SmallSTBlock(nn.Module):
    def __init__(self, Kt, Ks, n_vertex, last_block_channel, channels, act_func, graph_conv_type, gso, bias, droprate):
        super(SmallSTBlock, self).__init__()

        self.tmp_conv1 = TemporalConvLayer(Kt, last_block_channel, channels[0], n_vertex, act_func)
        self.graph_conv = GraphConvLayer(graph_conv_type, channels[0], channels[1], Ks, gso, bias)
        self.tmp_conv2 = TemporalConvLayer(Kt, channels[1], channels[2], n_vertex, act_func)
        
        self.temporal_attention = SmallTemporalAttention(channels[2], n_vertex)

        self.tc2_ln = nn.LayerNorm([n_vertex, channels[2]], eps=1e-12)        
        self.lr_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x = self.tmp_conv1(x)
        x = self.graph_conv(x)
        x = F.ReLU(x)  # ELU instead of ReLU for better gradient flow
        x = self.tmp_conv2(x)
            
        x = self.temporal_attention(x)
        x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.dropout(x)
            
        return x * self.lr_scale 

class STConvBlockTwoSTBlocks(nn.Module):
    def __init__(self, Kt, Ks, n_vertex, last_block_channel, channels, act_func, graph_conv_type, gso, bias, droprate):
        super(STConvBlockTwoSTBlocks, self).__init__()
        self.gso = gso
        self.tmp_conv1 = TemporalConvLayer(Kt, last_block_channel, channels[0], n_vertex, act_func)
        self.graph_conv = GraphConvLayer(graph_conv_type, channels[0], channels[1], Ks, gso, bias)
        self.tmp_conv2 = TemporalConvLayer(Kt, channels[1], channels[2], n_vertex, act_func)

        print(f"channels[0]: {channels[2]}")
        print(f"channels[1]: {channels[2]}")
        print(f"channels[2]: {channels[2]}")
        
        self.attention = nn.Sequential(
            nn.Conv2d(channels[2], channels[2], kernel_size=1),
            nn.Sigmoid()
        )
        
        self.tc2_ln = nn.LayerNorm([n_vertex, channels[2]], eps=1e-12)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x):
        #print(f"STBlock input shape: {x.shape}")
        x = self.tmp_conv1(x)
        #print(f"After first temporal conv: {x.shape}")
        x = self.graph_conv(x)
        #print(f"After graph conv: {x.shape}")
        x = self.elu(x)
        x = self.tmp_conv2(x)
        #print(f"After second temporal conv: {x.shape}")
        
        x = x * self.attention(x)
        
        x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.dropout(x)
        
        return x