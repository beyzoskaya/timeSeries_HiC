import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
import os
import seaborn as sns
from node2vec import Node2Vec
from scipy.stats import pearsonr, spearmanr
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from create_graph_and_embeddings_STGCN import *
import math

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
        if act_func == 'glu' or act_func == 'gtu':
            self.causal_conv = CausalConv2d(in_channels=c_in, out_channels=2 * c_out, kernel_size=(Kt, 1), enable_padding=False, dilation=1)
        else:
            self.causal_conv = CausalConv2d(in_channels=c_in, out_channels=c_out, kernel_size=(Kt, 1), enable_padding=False, dilation=1)
        self.relu = nn.ReLU()
        self.silu = nn.SiLU()
        self.act_func = act_func

    def forward(self, x):   
        x_in = self.align(x)[:, :, self.Kt - 1:, :]
        x_causal_conv = self.causal_conv(x)

        if self.act_func == 'glu' or self.act_func == 'gtu':
            x_p = x_causal_conv[:, : self.c_out, :, :]
            x_q = x_causal_conv[:, -self.c_out:, :, :]

            if self.act_func == 'glu':
                x = torch.mul((x_p + x_in), torch.sigmoid(x_q))
            else:
                x = torch.mul(torch.tanh(x_p + x_in), torch.sigmoid(x_q))
        elif self.act_func == 'relu':
            x = self.relu(x_causal_conv + x_in)
        elif self.act_func == 'silu':
            x = self.silu(x_causal_conv + x_in)
        else:
            raise NotImplementedError(f'ERROR: The activation function {self.act_func} is not implemented.')
        
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
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
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
        
        return cheb_graph_conv

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

    def forward(self, x):
        x_gc_in = self.align(x)
        if self.graph_conv_type == 'cheb_graph_conv':
            x_gc = self.cheb_graph_conv(x_gc_in)
        x_gc = x_gc.permute(0, 3, 1, 2)
        x_gc_out = torch.add(x_gc, x_gc_in)
        return x_gc_out

class STConvBlock(nn.Module):
    def __init__(self, Kt, Ks, n_vertex, last_block_channel, channels, act_func, graph_conv_type, gso, bias, droprate):
        super(STConvBlock, self).__init__()
        self.tmp_conv1 = TemporalConvLayer(Kt, last_block_channel, channels[0], n_vertex, act_func)
        self.graph_conv = GraphConvLayer(graph_conv_type, channels[0], channels[1], Ks, gso, bias)
        self.tmp_conv2 = TemporalConvLayer(Kt, channels[1], channels[2], n_vertex, act_func)
        self.tc2_ln = nn.LayerNorm([n_vertex, channels[2]], eps=1e-12)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x):
        x = self.tmp_conv1(x)
        x = self.graph_conv(x)
        x = self.relu(x)
        x = self.tmp_conv2(x)
        x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.dropout(x)
        return x

class OutputBlock(nn.Module):
    def __init__(self, Ko, last_block_channel, channels, end_channel, n_vertex, act_func, bias, droprate):
        super(OutputBlock, self).__init__()
        self.tmp_conv1 = TemporalConvLayer(Ko, last_block_channel, channels[0], n_vertex, act_func)
        self.fc1 = nn.Linear(in_features=channels[0], out_features=channels[1], bias=bias)
        self.fc2 = nn.Linear(in_features=channels[1], out_features=end_channel, bias=bias)
        self.tc1_ln = nn.LayerNorm([n_vertex, channels[0]], eps=1e-12)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x):
        x = self.tmp_conv1(x)
        x = self.tc1_ln(x.permute(0, 2, 3, 1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x).permute(0, 3, 1, 2)
        return x

class STGCNChebGraphConv(nn.Module):
    def __init__(self, Kt, Ks, n_vertex, blocks, act_func, graph_conv_type, gso, enable_bias, droprate, n_his):
        super(STGCNChebGraphConv, self).__init__()
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(STConvBlock(Kt, Ks, n_vertex, blocks[l][-1], blocks[l+1], act_func, graph_conv_type, gso, enable_bias, droprate))
        self.st_blocks = nn.Sequential(*modules)
        Ko = n_his - (len(blocks) - 3) * 2 * (Kt - 1)
        self.Ko = Ko
        if self.Ko > 1:
            self.output = OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex, act_func, enable_bias, droprate)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], bias=enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], bias=enable_bias)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=droprate)

    def forward(self, x):
        x = self.st_blocks(x)
        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)
        return x

def process_batch_for_stgcn(seq, label, device='cpu'):

    x_seq = torch.stack([g.x for g in seq])  # [seq_len, num_nodes, features]
    x_seq = x_seq.permute(2, 0, 1).unsqueeze(0)  # [1, features, seq_len, num_nodes]
    
    # Target: extract expression values (last feature)
    target_features = torch.stack([g.x for g in label])  # [pred_len, num_nodes, features]
    target = target_features[:, :, -1]  # [pred_len, num_nodes] - expression values
    
    return x_seq.to(device), target.to(device), seq

def create_graph_signal_operator(dataset, device='cpu'):

    adj_matrix = nx.adjacency_matrix(dataset.base_graph, nodelist=list(dataset.node_map.keys())).toarray()
    
    # Add self-loops
    adj_matrix = adj_matrix + np.eye(adj_matrix.shape[0])
    
    # Normalize adjacency matrix (symmetric normalization)
    degree = np.sum(adj_matrix, axis=1)
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.
    degree_matrix_inv_sqrt = np.diag(degree_inv_sqrt)
    
    # Normalized adjacency matrix
    norm_adj = degree_matrix_inv_sqrt @ adj_matrix @ degree_matrix_inv_sqrt
    
    # Convert to PyTorch tensor
    gso = torch.FloatTensor(norm_adj).to(device)
    
    return gso

def train_stgcn(model, dataset, epochs=10, learning_rate=1e-3, val_ratio=0.2, device='cpu'):
    model.to(device)
    model.train()

    sequences, labels = dataset.get_temporal_sequences()

    train_idx = torch.tensor([22, 9, 32, 15, 0, 3, 8, 18, 14, 13, 38, 2, 7, 4, 23, 37, 27, 29, 35, 17, 19, 25, 6, 21, 12, 10, 16, 39, 24, 33, 11, 34])
    val_idx = torch.tensor([28, 20, 26, 31, 30, 36, 1, 5])

    #train_sequences, train_labels, val_sequences, val_labels, _, _ = dataset.split_sequences_from_idx(sequences, labels, train_idx, val_idx)
    train_sequences, train_labels, val_sequences, val_labels, _, _ = dataset.split_sequences(sequences, labels)
    
    print(f"Total sequences: {len(sequences)}")
    print(f"Train sequences: {len(train_sequences)}")
    print(f"Validation sequences: {len(val_sequences)}")
   
    if len(train_sequences) > 0:
        x_sample, target_sample, _ = process_batch_for_stgcn(train_sequences[0], train_labels[0], device)
        print(f"Input shape for STGCN: {x_sample.shape}")  # Should be [1, features, seq_len, num_nodes]
        print(f"Target shape: {target_sample.shape}")  # Should be [pred_len, num_nodes]
    
    optimizer = optim.Adam(model.parameters(), lr=0.0007, weight_decay=1e-5)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for seq_graphs, label_graphs in zip(train_sequences, train_labels):
            optimizer.zero_grad()
            
            x_input, y_true, seq_graphs_list = process_batch_for_stgcn(seq_graphs, label_graphs, device)
            
            output = model(x_input)

            # Process STGCN output properly
            if output.dim() == 4:
                output_processed = output[:, :, -1, :].mean(dim=1)
            elif output.dim() == 3:
                output_processed = output[:, -1, :]

            if y_true.dim() == 2 and y_true.shape[0] > 1:
                y_true = y_true[-1:, :]

            # Prepare for loss function
            input_for_loss = x_input
            output_for_loss = output_processed.unsqueeze(1).unsqueeze(2)
            target_for_loss = y_true.unsqueeze(1).unsqueeze(2)

            loss = miRNA_enhanced_temporal_loss_stgcn(output_for_loss, target_for_loss, input_for_loss)
            
            # ADD THESE MISSING LINES:
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_train_loss = epoch_loss / len(train_sequences)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seq_graphs, label_graphs in zip(val_sequences, val_labels):
                x_input, y_true, seq_graphs_list = process_batch_for_stgcn(seq_graphs, label_graphs, device)

                output = model(x_input)

                if output.dim() == 4:
                    output_processed = output[:, :, -1, :].mean(dim=1)
                elif output.dim() == 3:
                    output_processed = output[:, -1, :]
                    
                if y_true.dim() == 2 and y_true.shape[0] > 1:
                    y_true = y_true[-1:, :]

                input_for_loss = x_input
                output_for_loss = output_processed.unsqueeze(1).unsqueeze(2)
                target_for_loss = y_true.unsqueeze(1).unsqueeze(2)

                loss = miRNA_enhanced_temporal_loss_stgcn(output_for_loss, target_for_loss, input_for_loss)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_sequences)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
    
    return model, val_sequences, val_labels, train_losses, val_losses, train_sequences, train_labels

def miRNA_enhanced_temporal_loss_stgcn(output, target, input_sequence, alpha=0.3, beta=0.2, gamma=0.3, delta=0.2):

    l1_loss = F.l1_loss(output, target)
 
    # Extract input expressions - STGCN format: [1, features, seq_len, num_nodes]
    # Take only the last feature (expression values) from the input
    input_expressions = input_sequence[:, -1:, :, :]  # [1, 1, seq_len, num_nodes]
    last_input = input_expressions[:, :, -1:, :]  # [1, 1, 1, num_nodes] - last time step
   
    output_reshaped = output.squeeze(1).squeeze(1)  # [1, num_nodes]
    target_reshaped = target.squeeze(1).squeeze(1)  # [1, num_nodes]
    last_input_reshaped = last_input.squeeze(1).squeeze(2)  # [1, num_nodes]

    # Direction loss
    true_change = target_reshaped - last_input_reshaped
    pred_change = output_reshaped - last_input_reshaped

    true_norm = F.normalize(true_change, p=2, dim=-1)
    pred_norm = F.normalize(pred_change, p=2, dim=-1)
    direction_cosine = torch.sum(true_norm * pred_norm, dim=-1)
    direction_loss = 1 - torch.mean(direction_cosine)
    
    scaled_direction_loss = direction_loss * 0.01
    
    def enhanced_trend_correlation(pred, target, sequence_expr):
        # sequence_expr: [1, 1, seq_len, num_nodes] -> [1, num_nodes, seq_len]
        sequence_expr_2d = sequence_expr.squeeze(1).permute(0, 2, 1)  # [1, num_nodes, seq_len]
        
        # Concatenate along time dimension
        pred_trend = torch.cat([sequence_expr_2d, pred.unsqueeze(2)], dim=2)  # [1, num_nodes, seq_len+1]
        target_trend = torch.cat([sequence_expr_2d, target.unsqueeze(2)], dim=2)  # [1, num_nodes, seq_len+1]

        def correlation_loss(x, y):
            x_centered = x - x.mean(dim=2, keepdim=True)
            y_centered = y - y.mean(dim=2, keepdim=True)
            x_norm = torch.sqrt(torch.sum(x_centered**2, dim=2) + 1e-8)
            y_norm = torch.sqrt(torch.sum(y_centered**2, dim=2) + 1e-8)
            correlation = torch.sum(x_centered * y_centered, dim=2) / (x_norm * y_norm + 1e-8)
            return 1 - correlation.mean()
        
        corr_loss = correlation_loss(pred_trend, target_trend)
        smoothness_loss = torch.mean(torch.abs(torch.diff(pred_trend, dim=2)))
       
        return corr_loss + 0.15 * smoothness_loss
 
    temporal_loss = enhanced_trend_correlation(output_reshaped, target_reshaped, input_expressions)
    scaled_temporal_loss = temporal_loss * 0.1
    
    # Consistency loss
    consistency_loss = torch.mean(torch.abs(output_reshaped - last_input_reshaped))
    
    total_loss = (
        alpha * l1_loss +
        beta * scaled_direction_loss + 
        gamma * scaled_temporal_loss +
        delta * consistency_loss
    )

    print(f"\nSTGCN Loss Components:")
    print(f"L1 loss: {l1_loss.item():.4f}")
    print(f"Direction Loss: {direction_loss.item():.4f}")
    print(f"Temporal Loss: {temporal_loss.item():.4f}")
    print(f"Consistency Loss: {consistency_loss.item():.4f}")
   
    return total_loss

def evaluate_stgcn_performance(model, val_sequences, val_labels, dataset, device='cpu', save_dir='plottings_STGCN'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for seq, label in zip(val_sequences, val_labels):
            x_input, target, seq_graphs_list = process_batch_for_stgcn(seq, label, device)
            
            # STGCN forward pass and output processing
            output = model(x_input)
            if output.dim() == 4:
                output = output[:, :, -1, :].mean(dim=1)  # [1, num_nodes]
            elif output.dim() == 3:
                output = output[:, -1, :]  # [1, num_nodes]
            
            pred = output.cpu().numpy()  # [1, num_nodes]
            pred = pred.squeeze()  # [num_nodes]
            true = target.cpu().numpy()  # [1, num_nodes] or [pred_len, num_nodes]
            
            # Handle target dimensions
            if true.ndim == 2:
                true = true[-1, :]  # Take last prediction step -> [num_nodes]
            elif true.ndim == 1:
                true = true  # Already [num_nodes]
            
            # Handle prediction dimensions (additional safety check)
            if pred.ndim > 1:
                pred = pred.flatten()  # Ensure [num_nodes]
            
            all_predictions.append(pred)
            all_targets.append(true)

    predictions = np.array(all_predictions)  # [num_sequences, num_nodes]
    targets = np.array(all_targets)          # [num_sequences, num_nodes]
    
    print(f"Final predictions shape: {predictions.shape}")
    print(f"Final targets shape: {targets.shape}")
    
    # Use the same evaluation functions as other models
    overall_metrics = calculate_overall_metrics(predictions, targets)
    gene_metrics = calculate_gene_metrics(predictions, targets, dataset)
    temporal_metrics = calculate_temporal_metrics_detailly(predictions, targets, dataset)

    create_evaluation_plots(predictions, targets, dataset, save_dir)
    
    metrics = {
        'Overall': overall_metrics,
        'Gene': gene_metrics,
        'Temporal': temporal_metrics
    }
    
    return metrics

# Include all the evaluation functions from DCRNN version (same implementations)
def calculate_overall_metrics(predictions, targets):
    """Calculate overall expression prediction metrics."""
    metrics = {}
    
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    metrics['MSE'] = mean_squared_error(target_flat, pred_flat)
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    metrics['MAE'] = mean_absolute_error(target_flat, pred_flat)
    metrics['R2_Score'] = r2_score(target_flat, pred_flat)
    metrics['Pearson_Correlation'], _ = pearsonr(target_flat, pred_flat)
    
    return metrics

def calculate_gene_metrics(predictions, targets, dataset):
    """Calculate gene-specific metrics."""
    metrics = {}
    genes = list(dataset.node_map.keys())
    
    gene_correlations = []
    gene_rmse = []
    gene_spearman_correlations = []

    for gene_idx, gene in enumerate(genes):
        pred_gene = predictions[:, gene_idx]
        true_gene = targets[:, gene_idx]
        
        if np.std(pred_gene) == 0 or np.std(true_gene) == 0:
            corr = 0.0
            spearman_corr = 0.0
        else:
            try:
                corr, _ = pearsonr(pred_gene, true_gene)
                spearman_corr, spearman_p = spearmanr(pred_gene, true_gene)
            except:
                corr = 0.0
                spearman_corr = 0.0
        
        rmse = np.sqrt(mean_squared_error(true_gene, pred_gene))
        
        gene_correlations.append((gene, corr))
        gene_spearman_correlations.append((gene, spearman_corr))
        gene_rmse.append(rmse)
    
    gene_correlations.sort(key=lambda x: x[1], reverse=True)
    
    metrics['Mean_Correlation'] = np.mean([corr for _, corr in gene_correlations])
    metrics['Mean_Spearman_Correlation'] = np.mean([corr for _, corr in gene_spearman_correlations])
    metrics['Best_Genes_Pearson'] = [gene for gene, _ in gene_correlations[:5]]
    metrics['Best_Genes_Spearman'] = [gene for gene, _ in gene_spearman_correlations[:5]]
    metrics['Gene_RMSE'] = {gene: rmse for gene, rmse in zip(genes, gene_rmse)}
    
    return metrics

def calculate_temporal_metrics_detailly(predictions, targets, dataset):
    """Calculate temporal prediction metrics."""
    metrics = {}
    
    def time_lagged_correlation(true_seq, pred_seq, max_lag=3):
        correlations = []
        for lag in range(max_lag + 1):
            if lag == 0:
                corr = np.corrcoef(true_seq, pred_seq)[0, 1]
            else:
                corr = np.corrcoef(true_seq[lag:], pred_seq[:-lag])[0, 1]
            correlations.append(corr)
        return np.max(correlations)
    
    def dtw_distance(true_seq, pred_seq):
        n, m = len(true_seq), len(pred_seq)
        dtw_matrix = np.zeros((n+1, m+1))
        dtw_matrix[1:, 0] = np.inf
        dtw_matrix[0, 1:] = np.inf
        
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = abs(true_seq[i-1] - pred_seq[j-1])
                dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],
                                            dtw_matrix[i, j-1],
                                            dtw_matrix[i-1, j-1])
        return dtw_matrix[n, m]
    
    genes = list(dataset.node_map.keys())
    temporal_metrics = []
    dtw_distances = []
    direction_accuracies = []
    
    for gene_idx, gene in enumerate(genes):
        true_seq = targets[:, gene_idx]
        pred_seq = predictions[:, gene_idx]
        
        temp_corr = time_lagged_correlation(true_seq, pred_seq)
        temporal_metrics.append(temp_corr)
        
        dtw_dist = dtw_distance(true_seq, pred_seq)
        dtw_distances.append(dtw_dist)
        
        true_changes = np.diff(true_seq)
        pred_changes = np.diff(pred_seq)
        dir_acc = np.mean(np.sign(true_changes) == np.sign(pred_changes))
        direction_accuracies.append(dir_acc)
    
    metrics['Mean_Temporal_Correlation'] = np.mean(temporal_metrics)
    metrics['Mean_DTW_Distance'] = np.mean(dtw_distances)
    metrics['Mean_Direction_Accuracy'] = np.mean(direction_accuracies)
    
    true_changes = np.diff(targets, axis=0)
    pred_changes = np.diff(predictions, axis=0)
    
    metrics['Mean_True_Change'] = np.mean(np.abs(true_changes))
    metrics['Mean_Pred_Change'] = np.mean(np.abs(pred_changes))
    metrics['Change_Magnitude_Ratio'] = metrics['Mean_Pred_Change'] / metrics['Mean_True_Change']
    
    return metrics

def create_evaluation_plots(predictions, targets, dataset, save_dir):
    """Create comprehensive evaluation plots."""
    plt.figure(figsize=(10, 8))
    plt.scatter(targets.flatten(), predictions.flatten(), alpha=0.1)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
    plt.xlabel('True Expression')
    plt.ylabel('Predicted Expression')
    plt.title('Expression Prediction Performance')
    plt.savefig(f'{save_dir}/overall_scatter.png')
    plt.close()
    
    true_changes = np.diff(targets, axis=0).flatten()
    pred_changes = np.diff(predictions, axis=0).flatten()
    
    plt.figure(figsize=(12, 6))
    plt.hist(true_changes, bins=50, alpha=0.5, label='Actual Changes')
    plt.hist(pred_changes, bins=50, alpha=0.5, label='Predicted Changes')
    plt.xlabel('Expression Change')
    plt.ylabel('Frequency')
    plt.title('Distribution of Expression Changes')
    plt.legend()
    plt.savefig(f'{save_dir}/change_distribution.png')
    plt.close()

def get_stgcn_predictions_and_targets(model, val_sequences, val_labels, dataset, device='cpu'):
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for seq, label in zip(val_sequences, val_labels):
            x_input, target, seq_graphs_list = process_batch_for_stgcn(seq, label, device)
            
            output = model(x_input)
            if output.dim() == 4:
                output = output[:, :, -1, :].mean(dim=1)  # [1, num_nodes]
            elif output.dim() == 3:
                output = output[:, -1, :]  # [1, num_nodes]
            
            pred = output.cpu().numpy()  # [1, num_nodes]
            pred = pred.squeeze()  # [num_nodes]
            true = target.cpu().numpy()  # [1, num_nodes] or [pred_len, num_nodes]
            
            # Handle target dimensions
            if true.ndim == 2:
                true = true[-1, :]  # Take last prediction step -> [num_nodes]
            elif true.ndim == 1:
                true = true  # Already [num_nodes]
            
            # Handle prediction dimensions (additional safety check)
            if pred.ndim > 1:
                pred = pred.flatten()  # Ensure [num_nodes]
            
            all_predictions.append(pred.reshape(1, -1))
            all_targets.append(true.reshape(1, -1))

    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")
    
    return predictions, targets

def plot_stgcn_predictions_train_val(model, train_sequences, train_labels, val_sequences, val_labels, dataset, device='cpu', save_dir='plottings_STGCN', genes_per_page=12):

    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # Process training data
    train_predictions = []
    train_targets = []
    
    with torch.no_grad():
        for seq, label in zip(train_sequences, train_labels):
            x_input, target, seq_graphs_list = process_batch_for_stgcn(seq, label, device)
            
            output = model(x_input)
            if output.dim() == 4:
                output = output[:, :, -1, :].mean(dim=1)  # [1, num_nodes]
            elif output.dim() == 3:
                output = output[:, -1, :]  # [1, num_nodes]
            
            pred = output.cpu().numpy()  # [1, num_nodes]
            pred = pred.squeeze()  # [num_nodes]
            true = target.cpu().numpy()  # [1, num_nodes] or [pred_len, num_nodes]
            
            # Handle target dimensions
            if true.ndim == 2:
                true = true[-1, :]  # Take last prediction step -> [num_nodes]
            elif true.ndim == 1:
                true = true  # Already [num_nodes]
            
            # Handle prediction dimensions (additional safety check)
            if pred.ndim > 1:
                pred = pred.flatten()  # Ensure [num_nodes]
                
            train_predictions.append(pred)
            train_targets.append(true)
        
        # Process validation data  
        val_predictions = []
        val_targets = []
        
        for seq, label in zip(val_sequences, val_labels):
            x_input, target, seq_graphs_list = process_batch_for_stgcn(seq, label, device)
            
            output = model(x_input)
            if output.dim() == 4:
                output = output[:, :, -1, :].mean(dim=1)  # [1, num_nodes]
            elif output.dim() == 3:
                output = output[:, -1, :]  # [1, num_nodes]
            
            pred = output.cpu().numpy()  # [1, num_nodes]
            pred = pred.squeeze()  # [num_nodes]
            true = target.cpu().numpy()  # [1, num_nodes] or [pred_len, num_nodes]
            
            # Handle target dimensions
            if true.ndim == 2:
                true = true[-1, :]  # Take last prediction step -> [num_nodes]
            elif true.ndim == 1:
                true = true  # Already [num_nodes]
            
            # Handle prediction dimensions (additional safety check)
            if pred.ndim > 1:
                pred = pred.flatten()  # Ensure [num_nodes]

            val_predictions.append(pred)
            val_targets.append(true)
    
    # Convert to arrays
    train_predictions = np.array(train_predictions)
    train_targets = np.array(train_targets)
    val_predictions = np.array(val_predictions)
    val_targets = np.array(val_targets)
    
    # Plot
    gene_names = list(dataset.node_map.keys())
    num_genes = dataset.num_nodes
    num_pages = (num_genes + genes_per_page - 1) // genes_per_page

    for page in range(num_pages):
        plt.figure(figsize=(20, 15))
        
        start_idx = page * genes_per_page
        end_idx = min((page + 1) * genes_per_page, num_genes)
        page_genes = gene_names[start_idx:end_idx]
        
        for i, gene_name in enumerate(page_genes):
            gene_idx = start_idx + i
            rows = (genes_per_page + 1) // 2
            plt.subplot(rows, 2, i + 1) 
        
            train_time_points = range(len(train_predictions))
            plt.plot(train_time_points, train_targets[:, gene_idx], label='Train Actual', color='blue', marker='o')
            plt.plot(train_time_points, train_predictions[:, gene_idx], label='Train Predicted', color='red', linestyle='--', marker='x')
     
            val_time_points = range(len(train_predictions), len(train_predictions) + len(val_predictions))
            plt.plot(val_time_points, val_targets[:, gene_idx], label='Val Actual', color='green', marker='o')
            plt.plot(val_time_points, val_predictions[:, gene_idx], label='Val Predicted', color='orange', linestyle='--', marker='x')
            
            plt.title(f'Gene: {gene_name}', fontsize=16)
            plt.xlabel('Time Points', fontsize=14)
            plt.ylabel('Expression Value', fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(loc='center left', bbox_to_anchor=(-0.3, 0.5), fontsize=12, frameon=False)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/gene_predictions_page_{page + 1}.pdf', dpi=900)
        plt.close()

def analyze_gene_characteristics(dataset, predictions, targets):
    """Analyze relationship between gene properties and prediction performance"""
    genes = list(dataset.node_map.keys())
    
    gene_correlations = {}
    for gene in genes:
        gene_idx = dataset.node_map[gene]
        pred_gene = predictions[:, gene_idx]
        true_gene = targets[:, gene_idx]
        corr, _ = pearsonr(pred_gene, true_gene)
        gene_correlations[gene] = corr
    
    gene_stats = {gene: {
        'degree': len(dataset.base_graph[gene]),
        'expression_range': None,
        'expression_std': None,
        'correlation': gene_correlations[gene]
    } for gene in genes}
    
    for gene in genes:
        all_expressions = []
        for t in dataset.time_points:
            gene1_expr = dataset.df[dataset.df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
            gene2_expr = dataset.df[dataset.df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
            expr_values = np.concatenate([gene1_expr, gene2_expr])
            all_expressions.extend(expr_values)
        
        all_expressions = np.array(all_expressions)
        gene_stats[gene].update({
            'expression_range': np.ptp(all_expressions),
            'expression_std': np.std(all_expressions)
        })
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    degrees = [gene_stats[gene]['degree'] for gene in genes]
    correlations = [gene_stats[gene]['correlation'] for gene in genes]
    plt.scatter(degrees, correlations)
    plt.xlabel('Number of Interactions')
    plt.ylabel('Prediction Correlation')
    plt.title('Gene Connectivity vs Prediction Performance')
    
    plt.subplot(2, 2, 2)
    ranges = [gene_stats[gene]['expression_range'] for gene in genes]
    plt.scatter(ranges, correlations)
    plt.xlabel('Expression Range')
    plt.ylabel('Prediction Correlation')
    plt.title('Expression Variability vs Prediction Performance')
    
    plt.subplot(2, 2, 3)
    plt.hist(correlations, bins=20)
    plt.xlabel('Correlation')
    plt.ylabel('Count')
    plt.title('Distribution of Gene Correlations')
    
    plt.tight_layout()
    plt.savefig('plottings_STGCN/gene_analysis.png')
    plt.close()
    
    print("\nGene Analysis Summary:")
    print("\nTop 5 Most Connected Genes:")
    sorted_by_degree = sorted(gene_stats.items(), key=lambda x: x[1]['degree'], reverse=True)[:5]
    for gene, stats in sorted_by_degree:
        print(f"{gene}: {stats['degree']} connections, correlation: {stats['correlation']:.4f}")
    
    print("\nTop 5 Most Variable Genes:")
    sorted_by_range = sorted(gene_stats.items(), key=lambda x: x[1]['expression_range'], reverse=True)[:5]
    for gene, stats in sorted_by_range:
        print(f"{gene}: range {stats['expression_range']:.4f}, correlation: {stats['correlation']:.4f}")
    
    print("\nTop 5 Best Predicted Genes:")
    sorted_by_corr = sorted(gene_stats.items(), key=lambda x: x[1]['correlation'], reverse=True)[:5]
    for gene, stats in sorted_by_corr:
        print(f"{gene}: correlation {stats['correlation']:.4f}, connections: {stats['degree']}")
    
    return gene_stats

def analyze_temporal_patterns(dataset, predictions, targets):
    time_points = dataset.time_points
    genes = list(dataset.node_map.keys())

    temporal_stats = {
        'prediction_lag': [],
        'pattern_complexity': [],
        'prediction_accuracy': []
    }

    time_point_accuracy = []
    for t in range(len(predictions)):
        corr = pearsonr(predictions[t].flatten(), targets[t].flatten())[0]
        time_point_accuracy.append(corr)
    
    plt.figure(figsize=(15, 5))
    plt.plot(time_point_accuracy)
    plt.xlabel('Time Point')
    plt.ylabel('Prediction Accuracy')
    plt.title('Prediction Accuracy Over Time')
    plt.savefig(f'plottings_STGCN/pred_accuracy.png')

    print("\nTemporal Analysis:")
    print(f"Best predicted time point: {np.argmax(time_point_accuracy)}")
    print(f"Worst predicted time point: {np.argmin(time_point_accuracy)}")
    print(f"Mean accuracy: {np.mean(time_point_accuracy):.4f}")
    print(f"Std of accuracy: {np.std(time_point_accuracy):.4f}")
    
    return temporal_stats


if __name__ == "__main__":
    # Initialize dataset
    dataset = TemporalGraphDataset(
        csv_file='/Users/beyzakaya/Desktop/timeSeries_HiC/mapped/mRNA/enhanced_interactions_synthetic_simple_mRNA.csv',
        embedding_dim=32, 
        seq_len=5,
        pred_len=1
    )

    test_graph = dataset.get_pyg_graph(dataset.time_points[0])
    print(f"Test graph valid: {test_graph.validate()}")
    print(f"Has edge_attr: {hasattr(test_graph, 'edge_attr') and test_graph.edge_attr is not None}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gso = create_graph_signal_operator(dataset, device)
    print(f"Graph Signal Operator shape: {gso.shape}")
    
    blocks = [
    [32],           # Input
    [32, 16, 32],   # ST block 1 - smaller intermediate
    [32, 16, 32],   # ST block 2 - consistent size
    [32, 16],       # Pre-output
    [1]             # Output
]

    model = STGCNChebGraphConv(
        Kt=2,                 
        Ks=2,                 
        n_vertex=dataset.num_nodes,
        blocks=blocks,
        act_func='glu',
        graph_conv_type='cheb_graph_conv',
        gso=gso,
        enable_bias=True,
        droprate=0.1,
        n_his=dataset.seq_len    # Historical sequence length (3)
    ).to(device)
    
    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB") 

    print(f"STGCN Model initialized with:")
    print(f"- Node features: {dataset.embedding_dim}")
    print(f"- Sequence length: {dataset.seq_len}")
    print(f"- Prediction length: {dataset.pred_len}")
    print(f"- Number of nodes: {dataset.num_nodes}")
    print(f"- Temporal kernel (Kt): 3")
    print(f"- Spatial kernel (Ks): 3")
    print(f"- Device: {device}")
 
    trained_model, val_sequences, val_labels, train_losses, val_losses, train_sequences, train_labels = train_stgcn(
        model=model,
        dataset=dataset,
        epochs=20,
        learning_rate=1e-4,
        val_ratio=0.2,
        device=device
    )
    
    metrics = evaluate_stgcn_performance(trained_model, val_sequences, val_labels, dataset, device)
    
    plot_stgcn_predictions_train_val(trained_model, train_sequences, train_labels, val_sequences, val_labels, dataset, device)

    print("\nSTGCN Model Performance Summary:")
    print("\nOverall Metrics:")
    for metric, value in metrics['Overall'].items():
        print(f"{metric}: {value:.4f}")

    print("\nGene Performance:")
    print(f"Mean Gene Correlation: {metrics['Gene']['Mean_Correlation']:.4f}")
    print(f"Mean Spearman Correlation: {metrics['Gene']['Mean_Spearman_Correlation']:.4f}")
    print(f"Best Performing Genes Pearson: {', '.join(metrics['Gene']['Best_Genes_Pearson'])}")
    print(f"Best Performing Genes Spearman: {', '.join(metrics['Gene']['Best_Genes_Spearman'])}")

    print("\nTemporal Performance:")
    print(f"Time-lagged Correlation: {metrics['Temporal']['Mean_Temporal_Correlation']:.4f}")
    print(f"DTW Distance: {metrics['Temporal']['Mean_DTW_Distance']:.4f}")
    print(f"Direction Accuracy: {metrics['Temporal']['Mean_Direction_Accuracy']:.4f}")
    print(f"Change Magnitude Ratio: {metrics['Temporal']['Change_Magnitude_Ratio']:.4f}")

    predictions, targets = get_stgcn_predictions_and_targets(model, val_sequences, val_labels, dataset, device)
    gene_stats = analyze_gene_characteristics(dataset, predictions, targets)
    temporal_stats = analyze_temporal_patterns(dataset, predictions, targets)
   
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('STGCN Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(np.log(train_losses), label='Log Training Loss')
    plt.plot(np.log(val_losses), label='Log Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.title('STGCN Training and Validation Loss (Log Scale)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('plottings_STGCN/training_losses.png')
    plt.close()
    
    print(f"\nTraining completed. Results saved in 'plottings_STGCN' directory.")