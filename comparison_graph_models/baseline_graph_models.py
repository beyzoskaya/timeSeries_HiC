import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import TGCN
#from torch_geometric_temporal.nn.stgcn import STConv
from torch_geometric.nn import ChebConv
from model.layers import *
from model.models import * 

class BaselineGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(BaselineGCN, self).__init__()
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, out_dim)
        self.fc = nn.Linear(out_dim, 1)
        self.dropout = nn.Dropout(0.2)
        self.elu = nn.ELU()

    def forward(self, x, edge_index):
        """
        x: tensor [batch_size, embedding_dim, n_nodes]
        edge_index: [2, num_edges] tensor
        """
        batch_size, embedding_dim, n_nodes = x.size()
        # reshape: merge batch & nodes
        x = x.permute(0, 2, 1)  # [batch, n_nodes, embedding_dim]
        x = x.reshape(batch_size * n_nodes, embedding_dim)  # [batch * n_nodes, embedding_dim]

        # Create batch tensor: tells PyG which graph each node belongs to
        batch = torch.arange(batch_size, device=x.device).repeat_interleave(n_nodes)

        # Apply first GCN
        x = self.gcn1(x, edge_index)       # [batch * n_nodes, hidden_dim]
        x = self.elu(x)
        x = self.dropout(x)

        # Apply second GCN
        x = self.gcn2(x, edge_index)       # [batch * n_nodes, out_dim]
        x = self.elu(x)
        x = self.dropout(x)

        # Final linear layer for each node
        x = self.fc(x)  # [batch * n_nodes, 1]

        # Reshape back: [batch, n_nodes, 1]
        x = x.view(batch_size, n_nodes, 1)

        # Permute to [batch, 1, 1, n_nodes] to match your expected shape
        x = x.permute(0, 2, 1).unsqueeze(2)

        return x


class TGCNCell(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.gate = GCNConv(in_channels + hidden_channels, 2 * hidden_channels)
        self.update = GCNConv(in_channels + hidden_channels, hidden_channels)

    def forward(self, x, edge_index, h):
        if h is None:
            h = torch.zeros(x.size(0), self.update.out_channels, device=x.device)

        combined = torch.cat([x, h], dim=1)
        gates = torch.sigmoid(self.gate(combined, edge_index))
        z, r = gates.chunk(2, dim=1)
        combined_r = torch.cat([x, r * h], dim=1)
        h_tilde = torch.tanh(self.update(combined_r, edge_index))
        h_new = (1 - z) * h + z * h_tilde
        return h_new

class BaselineTGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.tgcn_cell = TGCNCell(in_channels, hidden_channels)
        self.out_layer = nn.Linear(hidden_channels, out_channels)

    def forward(self, seq_graphs):
        """
        seq_graphs: list of PyG Data objects, each has:
            x: [num_nodes, in_channels]
            edge_index: [2, num_edges]
        """
        h = None
        outputs = []

        for data in seq_graphs:
            h = self.tgcn_cell(data.x, data.edge_index, h)
            out = self.out_layer(h)
            outputs.append(out)

        # shape: [seq_len, num_nodes, out_channels]
        return torch.stack(outputs, dim=0)

class BaselineSTGCN(nn.Module):
    def __init__(self, args, blocks, n_vertex, in_channels):
        super(BaselineSTGCN, self).__init__()

        self.n_vertex = n_vertex
        self.Kt = args.Kt
        self.Ks = args.Ks
        self.graph_conv_type = args.graph_conv_type  # Save for print/debug

        modules = []
        last_block_channel = in_channels 

        for l in range(len(blocks)):
            in_ch = last_block_channel
            out_ch = blocks[l][-1]  # last channel in the block definition (or blocks[l][1] if 2 elems)
            modules.append(
                layers.STConvBlock(
                    Kt=args.Kt,
                    Ks=args.Ks,
                    n_vertex=n_vertex,
                    last_block_channel=in_ch,
                    channels=blocks[l],
                    act_func=args.act_func,
                    graph_conv_type=args.graph_conv_type,
                    gso=args.gso,
                    bias=args.enable_bias,
                    droprate=args.droprate
                )
            )
            last_block_channel = out_ch

        self.st_blocks = nn.Sequential(*modules)

        # Compute output temporal length after convolutions
        Ko = args.n_his - len(blocks) * 2 * (args.Kt - 1)
        self.Ko = Ko

        # Output fully connected layers
        if Ko > 1:
            self.output = nn.Sequential(
                nn.Conv2d(in_channels=last_block_channel, out_channels=64, kernel_size=(Ko, 1)),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1))
            )
        else:
            self.output = nn.Sequential(
                nn.Conv2d(in_channels=last_block_channel, out_channels=1, kernel_size=(1, 1))
            )

        print(f"graph_conv_type: {self.graph_conv_type}")

    def forward(self, x):
        # x shape: (batch_size, in_channels, n_his, n_vertex)
        x = self.st_blocks(x)
        x = self.output(x)  # shape: (batch_size, 1, ?, n_vertex)
        x = x.squeeze(1).mean(dim=1)  # shape: (batch_size, n_vertex)
        return x


class STGCN(nn.Module):
    def __init__(self, args, blocks, n_vertex):
        """
        args: args with attributes like Kt, Ks, act_func, graph_conv_type, gso, enable_bias, droprate, n_his
        blocks: list of lists, e.g., [[32,32,32], [32,48,48], [48,32,32], [32], [1]]
        n_vertex: number of graph nodes
        """
        super(STGCN, self).__init__()

        modules = []
        for l in range(len(blocks) - 2):
            modules.append(
                layers.STConvBlockTwoSTBlocks(
                    Kt=args.Kt,
                    Ks=args.Ks,
                    n_vertex=n_vertex,
                    last_block_channel=blocks[l][-1],
                    channels=blocks[l+1],
                    act_func=args.act_func,
                    graph_conv_type=args.graph_conv_type,
                    gso=args.gso,
                    bias=args.enable_bias,
                    droprate=args.droprate
                )
            )
        self.st_blocks = nn.Sequential(*modules)

        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko
        print(f"[DEBUG] Ko: {self.Ko}")

        if self.Ko > 1:
            # last_block_channel matching final st_block output channels:
            last_block_channel = blocks[-2][-1]  # e.g., 48
            self.output = layers.OutputBlock(
                self.Ko,
                last_block_channel,
                blocks[-2],        # e.g., [48,32,32]
                blocks[-1][0], # final output channels
                n_vertex,
                args.act_func,
                args.enable_bias,
                args.droprate
            )

        elif self.Ko == 0:
            self.fc1 = nn.Linear(
                in_features=blocks[-2][-1],
                out_features=blocks[-2][0],
                bias=args.enable_bias
            )
            self.fc2 = nn.Linear(
                in_features=blocks[-2][0],
                out_features=blocks[-1][0],
                bias=args.enable_bias
            )
            self.elu = nn.ELU()

        self.expression_fc = nn.Linear(blocks[-1][0], 1)

    def forward(self, x):
        # x shape: [batch, channels, time, nodes]
        print(f"[DEBUG] Input x shape: {x.shape}") # --> ([1, 32, 3, 50]) [batch, channels, time, nodes]

        # Apply spatial-temporal conv blocks
        x = self.st_blocks(x)
        print(f"[DEBUG] After ST blocks: {x.shape}") # --> ([1, 48, 3, 50]) [batch, channels, time, nodes]

        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            # flatten and fc
            x = self.fc1(x.permute(0, 2, 3, 1))  # [B, T, N, C]
            x = self.elu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)  # back to [B, C, T, N]

        # [batch, channels, time, nodes] → [batch, time, nodes, channels]
        x = x.permute(0, 2, 3, 1)

        # final prediction: reduce feature dim to 1
        x = self.expression_fc(x)  # [B, T, N, 1]

        # [B, T, N, 1] → [B, 1, T, N]
        x = x.permute(0, 3, 1, 2)

        return x
