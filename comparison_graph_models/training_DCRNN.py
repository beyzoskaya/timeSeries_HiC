import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import networkx as nx
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Step 1: Data Extraction and Preprocessing
class GeneExpressionDataExtractor:
    """Extract gene expression matrix and adjacency matrix from your CSV data"""
    
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.df = None
        self.node_map = {}
        self.time_points = []
        self.expression_matrix = None
        self.adjacency_matrix = None
        
    def load_and_preprocess_data(self):
        """Load CSV and extract basic information"""
        print("Loading and preprocessing data...")
        
        # Load data
        self.df = pd.read_csv(self.csv_file)
        self.df['Gene1_clean'] = self.df['Gene1']
        self.df['Gene2_clean'] = self.df['Gene2']
        self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed', case=False)]
        
        # Get unique genes
        unique_genes = pd.concat([self.df['Gene1_clean'], self.df['Gene2_clean']]).unique()
        self.node_map = {gene: idx for idx, gene in enumerate(unique_genes)}
        self.num_genes = len(self.node_map)
        
        # Extract time points
        self.time_cols = [col for col in self.df.columns if 'Time_' in col]
        self.time_points = sorted(list(set([float(col.split('_')[-1])  
                                          for col in self.time_cols if 'Gene1' in col])))
        
        # Remove time point 154.0 if exists (as in your original code)
        if 154.0 in self.time_points:
            self.time_points = [tp for tp in self.time_points if tp != 154.0]
            self.df = self.df.loc[:, ~self.df.columns.str.contains('Time_154.0', case=False)]
        
        print(f"Found {self.num_genes} genes and {len(self.time_points)} time points")
        print(f"Time points: {self.time_points}")

    def _investigate_data_distribution(self):
        """Investigate the distribution of expression data"""
        print("\n" + "="*50)
        print("INVESTIGATION 1: DATA DISTRIBUTION ANALYSIS")
        print("="*50)
        
        print(f"Expression matrix shape: {self.expression_matrix.shape}")
        print(f"Total values: {self.expression_matrix.size}")
        print(f"Non-zero values: {np.count_nonzero(self.expression_matrix)}")
        print(f"Zero values: {np.sum(self.expression_matrix == 0)}")
        print(f"Percentage non-zero: {100 * np.count_nonzero(self.expression_matrix) / self.expression_matrix.size:.2f}%")
        
        # Basic statistics
        print(f"\nExpression Value Statistics:")
        print(f"  Min: {self.expression_matrix.min():.6f}")
        print(f"  Max: {self.expression_matrix.max():.6f}")
        print(f"  Mean: {self.expression_matrix.mean():.6f}")
        print(f"  Std: {self.expression_matrix.std():.6f}")
        print(f"  Median: {np.median(self.expression_matrix):.6f}")
        
        # Per-gene analysis
        gene_means = np.mean(self.expression_matrix, axis=0)
        gene_stds = np.std(self.expression_matrix, axis=0)
        gene_zeros = np.sum(self.expression_matrix == 0, axis=0)
        
        print(f"\nPer-Gene Analysis:")
        print(f"  Genes with zero variance: {np.sum(gene_stds == 0)}")
        print(f"  Genes with very low variance (<0.01): {np.sum(gene_stds < 0.01)}")
        print(f"  Genes with all zeros: {np.sum(gene_zeros == len(self.time_points))}")
        print(f"  Average non-zero values per gene: {np.mean(len(self.time_points) - gene_zeros):.2f}")
        
        # Per-timepoint analysis
        time_means = np.mean(self.expression_matrix, axis=1)
        time_stds = np.std(self.expression_matrix, axis=1)
        
        print(f"\nPer-Timepoint Analysis:")
        print(f"  Timepoint mean range: {time_means.min():.4f} - {time_means.max():.4f}")
        print(f"  Timepoint std range: {time_stds.min():.4f} - {time_stds.max():.4f}")
        
        # Show some examples
        gene_names = list(self.node_map.keys())
        print(f"\nSample Gene Statistics:")
        for i in range(min(5, len(gene_names))):
            gene_name = gene_names[i]
            gene_idx = self.node_map[gene_name]
            print(f"  {gene_name}: mean={gene_means[gene_idx]:.4f}, "
                  f"std={gene_stds[gene_idx]:.4f}, zeros={gene_zeros[gene_idx]}")
        
        # Check for potential issues
        print(f"\nPotential Issues Detected:")
        if np.sum(gene_stds == 0) > 0:
            print(f"  ⚠️  {np.sum(gene_stds == 0)} genes have zero variance!")
        if np.sum(gene_stds < 0.01) > len(gene_names) * 0.1:
            print(f"  ⚠️  {np.sum(gene_stds < 0.01)} genes have very low variance!")
        if np.count_nonzero(self.expression_matrix) / self.expression_matrix.size < 0.5:
            print(f"  ⚠️  Data is very sparse ({100 * np.count_nonzero(self.expression_matrix) / self.expression_matrix.size:.1f}% non-zero)!")
        if self.expression_matrix.max() > 100 * self.expression_matrix.mean():
            print(f"  ⚠️  Large dynamic range detected (max/mean = {self.expression_matrix.max() / self.expression_matrix.mean():.1f})!")
        
    def extract_expression_matrix(self):
        """Extract expression matrix: (num_timepoints, num_genes)"""
        print("Extracting expression matrix...")
        
        # Initialize expression matrix
        self.expression_matrix = np.zeros((len(self.time_points), self.num_genes))
        print("Expression stats:")
        print(f"Min: {self.expression_matrix.min()}, Max: {self.expression_matrix.max()}")
        print(f"Mean: {self.expression_matrix.mean()}, Std: {self.expression_matrix.std()}")
        print(f"Genes with zero variance: {np.sum(np.std(self.expression_matrix, axis=0) == 0)}")
        
        # Create a mapping of all gene expressions at each timepoint
        gene_expressions = {}
        
        # Extract expressions from the pairwise data
        for _, row in self.df.iterrows():
            gene1 = row['Gene1_clean']
            gene2 = row['Gene2_clean']
            
            for t_idx, t in enumerate(self.time_points):
                gene1_expr_col = f'Gene1_Time_{t}'
                gene2_expr_col = f'Gene2_Time_{t}'
                
                if gene1_expr_col in row and not pd.isna(row[gene1_expr_col]):
                    gene_expressions[(gene1, t)] = row[gene1_expr_col]
                
                if gene2_expr_col in row and not pd.isna(row[gene2_expr_col]):
                    gene_expressions[(gene2, t)] = row[gene2_expr_col]
        
        # Fill expression matrix
        for (gene, timepoint), expression in gene_expressions.items():
            if gene in self.node_map:
                gene_idx = self.node_map[gene]
                time_idx = self.time_points.index(timepoint)
                self.expression_matrix[time_idx, gene_idx] = expression
        
        print(f"Expression matrix shape: {self.expression_matrix.shape}")
        print(f"Non-zero values: {np.count_nonzero(self.expression_matrix)}")

        print("Expression stats:")
        print(f"Min: {self.expression_matrix.min()}, Max: {self.expression_matrix.max()}")
        print(f"Mean: {self.expression_matrix.mean()}, Std: {self.expression_matrix.std()}")
        print(f"Genes with zero variance: {np.sum(np.std(self.expression_matrix, axis=0) == 0)}")

        self._investigate_data_distribution()
        
        return self.expression_matrix
    
    def _investigate_graph_structure(self):
        """Investigate the graph structure"""
        print("\n" + "="*50)
        print("INVESTIGATION 2: GRAPH STRUCTURE ANALYSIS")
        print("="*50)
        
        print(f"Adjacency matrix shape: {self.adjacency_matrix.shape}")
        
        # Remove self-loops for analysis
        adj_no_self = self.adjacency_matrix.copy()
        np.fill_diagonal(adj_no_self, 0)
        
        total_possible_edges = self.num_genes * (self.num_genes - 1) // 2
        actual_edges = np.count_nonzero(adj_no_self) // 2
        
        print(f"Total possible edges: {total_possible_edges}")
        print(f"Actual edges: {actual_edges}")
        print(f"Graph density: {actual_edges / total_possible_edges:.4f}")
        
        # Edge weight statistics
        edge_weights = adj_no_self[adj_no_self > 0]
        if len(edge_weights) > 0:
            print(f"\nEdge Weight Statistics:")
            print(f"  Min weight: {edge_weights.min():.6f}")
            print(f"  Max weight: {edge_weights.max():.6f}")
            print(f"  Mean weight: {edge_weights.mean():.6f}")
            print(f"  Std weight: {edge_weights.std():.6f}")
            print(f"  Median weight: {np.median(edge_weights):.6f}")
        
        # Node degree analysis
        node_degrees = np.sum(adj_no_self > 0, axis=1)
        print(f"\nNode Degree Analysis:")
        print(f"  Min degree: {node_degrees.min()}")
        print(f"  Max degree: {node_degrees.max()}")
        print(f"  Mean degree: {node_degrees.mean():.2f}")
        print(f"  Std degree: {node_degrees.std():.2f}")
        print(f"  Isolated nodes (degree=0): {np.sum(node_degrees == 0)}")
        
        # Check connectivity
        if actual_edges > 0:
            # Simple connectivity check
            G = nx.from_numpy_array(adj_no_self)
            n_components = nx.number_connected_components(G)
            largest_component_size = len(max(nx.connected_components(G), key=len))
            
            print(f"\nConnectivity Analysis:")
            print(f"  Number of connected components: {n_components}")
            print(f"  Largest component size: {largest_component_size}/{self.num_genes}")
            print(f"  Graph is connected: {n_components == 1}")
        
        # Show some high-weight edges
        if len(edge_weights) > 0:
            gene_names = list(self.node_map.keys())
            print(f"\nTop 5 Strongest Connections:")
            edge_indices = np.where(adj_no_self > 0)
            edge_weights_with_idx = [(adj_no_self[i, j], i, j) for i, j in zip(*edge_indices) if i < j]
            edge_weights_with_idx.sort(reverse=True)
            
            for k, (weight, i, j) in enumerate(edge_weights_with_idx[:5]):
                print(f"  {gene_names[i]} - {gene_names[j]}: {weight:.4f}")
        
        # Potential issues
        print(f"\nPotential Graph Issues:")
        if actual_edges == 0:
            print(f"  ⚠️  No edges in the graph!")
        elif actual_edges < self.num_genes - 1:
            print(f"  ⚠️  Graph might be disconnected!")
        if np.sum(node_degrees == 0) > 0:
            print(f"  ⚠️  {np.sum(node_degrees == 0)} isolated nodes detected!")
        if actual_edges / total_possible_edges < 0.1:
            print(f"  ⚠️  Very sparse graph (density = {actual_edges / total_possible_edges:.4f})!")
        if len(edge_weights) > 0 and edge_weights.std() / edge_weights.mean() > 2:
            print(f"  ⚠️  High variance in edge weights (CV = {edge_weights.std() / edge_weights.mean():.2f})!")

    
    def build_adjacency_matrix(self, use_expression=True):
        """Build adjacency matrix from graph relationships"""
        print("Building adjacency matrix...")
        
        # Initialize adjacency matrix
        self.adjacency_matrix = np.zeros((self.num_genes, self.num_genes))
        
        # Build graph relationships
        for _, row in self.df.iterrows():
            gene1 = row['Gene1_clean']
            gene2 = row['Gene2_clean']
            
            if gene1 in self.node_map and gene2 in self.node_map:
                i = self.node_map[gene1]
                j = self.node_map[gene2]
                
                # Calculate edge weight using your original logic
                hic_weight = row['HiC_Interaction'] if not pd.isna(row['HiC_Interaction']) else 0
                compartment_sim = 1 if row['Gene1_Compartment'] == row['Gene2_Compartment'] else 0
                tad_dist = abs(row['Gene1_TAD_Boundary_Distance'] - row['Gene2_TAD_Boundary_Distance'])
                tad_sim = 1 / (1 + tad_dist)
                ins_sim = 1 / (1 + abs(row['Gene1_Insulation_Score'] - row['Gene2_Insulation_Score']))
                
                if use_expression:
                    # Add expression similarity (average across all timepoints)
                    expr_sims = []
                    for t in self.time_points:
                        gene1_expr = row.get(f'Gene1_Time_{t}', 0.0)
                        gene2_expr = row.get(f'Gene2_Time_{t}', 0.0)
                        expr_sim = 1 / (1 + abs(gene1_expr - gene2_expr)) if gene1_expr != 0 and gene2_expr != 0 else 0
                        #expr_sims.append(expr_sim)
                    #avg_expr_sim = np.mean(expr_sims) if expr_sims else 0
                    
                    weight = (hic_weight * 0.25 +
                            compartment_sim * 0.1 +
                            tad_sim * 0.1 +
                            ins_sim * 0.1 +
                            expr_sim * 0.45)
                else:
                    weight = (hic_weight * 0.4 + 
                            compartment_sim * 0.2 + 
                            tad_sim * 0.2 + 
                            ins_sim * 0.2)
                
                # Make symmetric
                self.adjacency_matrix[i, j] = weight
                self.adjacency_matrix[j, i] = weight
        
        row_sums = self.adjacency_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self.adjacency_matrix = self.adjacency_matrix / row_sums
        
        # Add self-loops
        np.fill_diagonal(self.adjacency_matrix, 1.0)

        self._investigate_graph_structure()
        
        print(f"Adjacency matrix shape: {self.adjacency_matrix.shape}")
        print(f"Number of edges: {np.count_nonzero(self.adjacency_matrix) // 2}")

        
        return self.adjacency_matrix

# Step 2: Dataset for DCRNN
class GeneExpressionSequenceDataset(Dataset):
    """Dataset for creating temporal sequences for DCRNN"""
    
    def __init__(self, expression_matrix: np.ndarray, seq_len: int = 3, pred_len: int = 1):
        self.expression_matrix = expression_matrix
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.sequences = self._create_sequences()
        
    def _create_sequences(self):
        """Create input-target sequences"""
        sequences = []
        num_timepoints, num_genes = self.expression_matrix.shape
        
        for i in range(num_timepoints - self.seq_len - self.pred_len + 1):
            # Input sequence: (seq_len, num_genes)
            input_seq = self.expression_matrix[i:i+self.seq_len]
            # Target sequence: (pred_len, num_genes)  
            target_seq = self.expression_matrix[i+self.seq_len:i+self.seq_len+self.pred_len]
            
            sequences.append((input_seq, target_seq))
        
        print(f"Created {len(sequences)} sequences")
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.sequences[idx]
        return (torch.FloatTensor(input_seq), torch.FloatTensor(target_seq))

# Step 3: DCRNN Implementation (Simplified)
class DiffusionConvolution(nn.Module):
    """Diffusion convolution for DCRNN"""
    
    def __init__(self, input_dim: int, output_dim: int, max_diffusion_step: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_diffusion_step = max_diffusion_step
        
        # Weight matrix for all diffusion steps
        total_input_dim = input_dim * (max_diffusion_step * 2 + 1)
        self.weight = nn.Parameter(torch.FloatTensor(total_input_dim, output_dim))
        self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, inputs: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (batch_size, num_nodes, input_dim)
            adj_matrix: (num_nodes, num_nodes)
        """
        batch_size, num_nodes, _ = inputs.shape
        
        # Normalize adjacency matrix
        adj_norm = self._normalize_adj(adj_matrix)
        
        # Collect diffusion outputs
        outputs = [inputs]  # 0-step diffusion
        
        # Forward diffusion
        x = inputs
        for _ in range(self.max_diffusion_step):
            x = torch.matmul(adj_norm, x)
            outputs.append(x)
        
        # Backward diffusion
        adj_T = adj_norm.transpose(0, 1)
        x = inputs
        for _ in range(self.max_diffusion_step):
            x = torch.matmul(adj_T, x)
            outputs.append(x)
        
        # Concatenate and apply linear transformation
        concatenated = torch.cat(outputs, dim=-1)
        output = torch.matmul(concatenated, self.weight) + self.bias
        
        return output
    
    def _normalize_adj(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        """Normalize adjacency matrix"""
        # Add small epsilon to avoid division by zero
        adj_matrix = adj_matrix + torch.eye(adj_matrix.size(0), device=adj_matrix.device) * 1e-6
        
        # Row normalization
        row_sum = torch.sum(adj_matrix, dim=1, keepdim=True)
        adj_normalized = adj_matrix / (row_sum + 1e-8)
        
        return adj_normalized

class DCGRUCell(nn.Module):
    """DCGRU Cell for DCRNN"""
    
    def __init__(self, input_dim: int, hidden_dim: int, max_diffusion_step: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Gates
        self.gate_conv = DiffusionConvolution(
            input_dim + hidden_dim, 2 * hidden_dim, max_diffusion_step
        )
        # Candidate
        self.candidate_conv = DiffusionConvolution(
            input_dim + hidden_dim, hidden_dim, max_diffusion_step
        )
    
    def forward(self, inputs: torch.Tensor, hidden_state: torch.Tensor, 
                adj_matrix: torch.Tensor) -> torch.Tensor:
        # Concatenate input and hidden state
        combined = torch.cat([inputs, hidden_state], dim=-1)
        
        # Compute gates
        gates = torch.sigmoid(self.gate_conv(combined, adj_matrix))
        reset_gate, update_gate = torch.split(gates, self.hidden_dim, dim=-1)
        
        # Apply reset gate
        reset_hidden = reset_gate * hidden_state
        
        # Compute candidate
        candidate_input = torch.cat([inputs, reset_hidden], dim=-1)
        candidate = torch.tanh(self.candidate_conv(candidate_input, adj_matrix))
        
        # Update hidden state
        new_hidden = update_gate * hidden_state + (1 - update_gate) * candidate
        
        return new_hidden

class DCRNN(nn.Module):
    """DCRNN Model for Gene Expression Prediction"""
    
    def __init__(self, num_genes: int, hidden_dim: int = 64, num_layers: int = 2, 
                 max_diffusion_step: int = 2, seq_len: int = 3, pred_len: int = 1):
        super().__init__()
        
        self.num_genes = num_genes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Input projection (gene expression to hidden dim)
        self.input_projection = nn.Linear(1, hidden_dim)
        
        # Encoder
        self.encoder_cells = nn.ModuleList()
        for i in range(num_layers):
            input_dim = hidden_dim if i == 0 else hidden_dim
            self.encoder_cells.append(DCGRUCell(input_dim, hidden_dim, max_diffusion_step))
        
        # Decoder
        self.decoder_cells = nn.ModuleList()
        for i in range(num_layers):
            input_dim = hidden_dim if i == 0 else hidden_dim
            self.decoder_cells.append(DCGRUCell(input_dim, hidden_dim, max_diffusion_step))
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, 1)
        
    def forward(self, inputs: torch.Tensor, adj_matrix: torch.Tensor, 
                targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            inputs: (batch_size, seq_len, num_genes)
            adj_matrix: (num_genes, num_genes)
            targets: (batch_size, pred_len, num_genes) for teacher forcing
        """
        batch_size, seq_len, num_genes = inputs.shape
        
        # Project inputs to hidden dimension
        # inputs: (batch_size, seq_len, num_genes) -> (batch_size, seq_len, num_genes, hidden_dim)
        inputs_projected = self.input_projection(inputs.unsqueeze(-1))
        
        # Initialize hidden states
        hidden_states = []
        for _ in range(self.num_layers):
            hidden_states.append(torch.zeros(batch_size, num_genes, self.hidden_dim, 
                                           device=inputs.device))
        
        # Encoder
        for t in range(seq_len):
            layer_input = inputs_projected[:, t, :, :]  # (batch_size, num_genes, hidden_dim)
            
            for layer in range(self.num_layers):
                if layer == 0:
                    cell_input = layer_input
                else:
                    cell_input = hidden_states[layer-1]
                
                hidden_states[layer] = self.encoder_cells[layer](
                    cell_input, hidden_states[layer], adj_matrix
                )
        
        # Decoder
        outputs = []
        decoder_input = torch.zeros(batch_size, num_genes, self.hidden_dim, device=inputs.device)
        
        for t in range(self.pred_len):
            # Teacher forcing during training
            if self.training and targets is not None:
                if t < targets.shape[1]:
                    decoder_input = self.input_projection(targets[:, t, :].unsqueeze(-1))
            
            for layer in range(self.num_layers):
                if layer == 0:
                    cell_input = decoder_input
                else:
                    cell_input = hidden_states[layer-1]
                
                hidden_states[layer] = self.decoder_cells[layer](
                    cell_input, hidden_states[layer], adj_matrix
                )
            
            # Generate output
            output = self.output_projection(hidden_states[-1]).squeeze(-1)  # (batch_size, num_genes)
            outputs.append(output)
            
            # Use output as next input
            decoder_input = self.input_projection(output.unsqueeze(-1))
        
        return torch.stack(outputs, dim=1)  # (batch_size, pred_len, num_genes)

# Step 4: Training and Evaluation
class DCRNNTrainer:
    """Trainer for DCRNN model"""
    
    def __init__(self, model: DCRNN, adj_matrix: np.ndarray, gene_names: List[str], 
                 device: torch.device = None):
        self.model = model
        self.adj_matrix = torch.FloatTensor(adj_matrix)
        self.gene_names = gene_names
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        self.adj_matrix = self.adj_matrix.to(self.device)
        
    def train_model(self, train_loader: DataLoader, 
                   num_epochs: int = 100, learning_rate: float = 0.0001, 
                   weight_decay: float = 1e-4):
        """Train the DCRNN model without validation set"""
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
        criterion = nn.MSELoss()
        
        train_losses = []
        
        print(f"Training on device: {self.device}")
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device)  # (batch_size, seq_len, num_genes)
                targets = targets.to(self.device)  # (batch_size, pred_len, num_genes)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs, self.adj_matrix, targets)
                loss = criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            scheduler.step()
            
            if epoch % 20 == 0:
                print(f'Epoch {epoch:03d}: Train Loss: {train_loss:.6f}')
        
        return train_losses
    
    def predict(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on test set"""
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for inputs, target_batch in test_loader:
                inputs = inputs.to(self.device)
                target_batch = target_batch.to(self.device)
                
                outputs = self.model(inputs, self.adj_matrix)
                
                predictions.append(outputs.cpu().numpy())
                targets.append(target_batch.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)

        print(f"Predictions shape: {predictions.shape}")
        print(f"Targets shape: {targets.shape}")

        print(f"Example predictions:\n{predictions[:5]}")
        print(f"Example targets:\n{targets[:5]}")
        
        return predictions, targets

# Step 5: Evaluation Metrics
def calculate_gene_metrics(predictions: np.ndarray, targets: np.ndarray, 
                          gene_names: List[str]) -> Dict:
    """
    Calculate gene-specific metrics for DCRNN predictions
    
    Args:
        predictions: (num_samples, pred_len, num_genes)
        targets: (num_samples, pred_len, num_genes)
        gene_names: list of gene names
    """
    metrics = {}
    
    # Flatten time dimension for per-gene analysis
    pred_flat = predictions.reshape(-1, predictions.shape[-1])  # (num_samples*pred_len, num_genes)
    target_flat = targets.reshape(-1, targets.shape[-1])
    
    gene_correlations = []
    gene_spearman_correlations = []
    gene_rmse = []
    
    for gene_idx, gene in enumerate(gene_names):
        pred_gene = pred_flat[:, gene_idx]
        true_gene = target_flat[:, gene_idx]
        
        # Skip genes with constant values
        if np.std(pred_gene) == 0 or np.std(true_gene) == 0:
            continue
            
        corr, _ = pearsonr(pred_gene, true_gene)
        spearman_corr, _ = spearmanr(pred_gene, true_gene)
        rmse = np.sqrt(mean_squared_error(true_gene, pred_gene))
        
        if not np.isnan(corr):
            gene_correlations.append((gene, corr))
        if not np.isnan(spearman_corr):
            gene_spearman_correlations.append((gene, spearman_corr))
        
        gene_rmse.append(rmse)
    
    # Sort genes by correlation
    gene_correlations.sort(key=lambda x: x[1], reverse=True)
    gene_spearman_correlations.sort(key=lambda x: x[1], reverse=True)
    
    metrics['Mean_Correlation'] = np.mean([corr for _, corr in gene_correlations]) if gene_correlations else 0
    metrics['Mean_Spearman_Correlation'] = np.mean([corr for _, corr in gene_spearman_correlations]) if gene_spearman_correlations else 0
    metrics['Best_Genes_Pearson'] = [gene for gene, _ in gene_correlations[:5]]
    metrics['Best_Genes_Spearman'] = [gene for gene, _ in gene_spearman_correlations[:5]]
    metrics['Worst_Genes_Pearson'] = [gene for gene, _ in gene_correlations[-5:]]
    metrics['Mean_RMSE'] = np.mean(gene_rmse)
    metrics['Num_Valid_Genes'] = len(gene_correlations)
    
    return metrics

def _calculate_baseline_metrics(targets, gene_names):
    """Calculate baseline metrics for comparison"""
    # Flatten targets for baseline calculation
    targets_flat = targets.reshape(-1, targets.shape[-1])
    
    # Baseline 1: Predict last known value
    last_value_preds = np.roll(targets_flat, 1, axis=0)  # Simple shift
    last_value_preds[0] = targets_flat[0]  # Handle first prediction
    
    # Baseline 2: Predict mean value for each gene
    gene_means = np.mean(targets_flat, axis=0)
    mean_preds = np.tile(gene_means, (targets_flat.shape[0], 1))
    
    # Calculate baseline correlations
    baseline_metrics = {}
    
    # Last value baseline
    last_val_corrs = []
    for gene_idx in range(len(gene_names)):
        if np.std(targets_flat[:, gene_idx]) > 0 and np.std(last_value_preds[:, gene_idx]) > 0:
            corr, _ = pearsonr(targets_flat[:, gene_idx], last_value_preds[:, gene_idx])
            if not np.isnan(corr):
                last_val_corrs.append(corr)
    
    # Mean baseline
    mean_corrs = []
    for gene_idx in range(len(gene_names)):
        if np.std(targets_flat[:, gene_idx]) > 0:
            corr, _ = pearsonr(targets_flat[:, gene_idx], mean_preds[:, gene_idx])
            if not np.isnan(corr):
                mean_corrs.append(corr)
    
    baseline_metrics['last_value_correlation'] = np.mean(last_val_corrs) if last_val_corrs else 0
    baseline_metrics['mean_value_correlation'] = np.mean(mean_corrs) if mean_corrs else 0
    
    return baseline_metrics

# Step 6: Complete Pipeline Function
def run_dcrnn_gene_expression_pipeline(csv_file: str, seq_len: int = 3, pred_len: int = 1, 
                                      hidden_dim: int = 32, num_layers: int = 2, 
                                      batch_size: int = 8, num_epochs: int = 100,
                                      test_split: float = 0.2):
    """
    Complete pipeline for DCRNN gene expression prediction (without validation set)
    """
    
    print("=" * 60)
    print("DCRNN Gene Expression Prediction Pipeline")
    print("=" * 60)
    
    # Step 1: Data Extraction
    extractor = GeneExpressionDataExtractor(csv_file)
    extractor.load_and_preprocess_data()
    expression_matrix = extractor.extract_expression_matrix()
    adjacency_matrix = extractor.build_adjacency_matrix(use_expression=False)
    
    # Normalize expression data
    #scaler = StandardScaler()
    #expression_matrix_scaled = scaler.fit_transform(expression_matrix)
    
    # Step 2: Create Dataset
    dataset = GeneExpressionSequenceDataset(expression_matrix, seq_len, pred_len)
    
    # Step 3: Train/Test Split (no validation set)
    dataset_size = len(dataset)
    test_size = int(dataset_size * test_split)
    train_size = dataset_size - test_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Step 4: Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DCRNN(
        num_genes=extractor.num_genes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        seq_len=seq_len,
        pred_len=pred_len
    )
    
    # Step 5: Training
    gene_names = list(extractor.node_map.keys())
    trainer = DCRNNTrainer(model, adjacency_matrix, gene_names, device)
    
    print(f"\nTraining DCRNN model...")
    train_losses = trainer.train_model(train_loader, num_epochs=num_epochs)
    
    # Step 6: Evaluation
    print(f"\nEvaluating on test set...")
    predictions, targets = trainer.predict(test_loader)
    
    # Calculate metrics
    metrics = calculate_gene_metrics(predictions, targets, gene_names)
    baseline_metrics = _calculate_baseline_metrics(targets, gene_names)
    
    # Print results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Mean Pearson Correlation: {metrics['Mean_Correlation']:.4f}")
    print(f"Mean Spearman Correlation: {metrics['Mean_Spearman_Correlation']:.4f}")
    print(f"Mean RMSE: {metrics['Mean_RMSE']:.4f}")
    print(f"Valid genes evaluated: {metrics['Num_Valid_Genes']}/{len(gene_names)}")
    print(f"\nTop 5 genes (Pearson): {metrics['Best_Genes_Pearson']}")
    print(f"Top 5 genes (Spearman): {metrics['Best_Genes_Spearman']}")
    print(f"Bottom 5 genes (Pearson): {metrics['Worst_Genes_Pearson']}")
    
    return {
        'model': model,
        'trainer': trainer,
        'metrics': metrics,
        'predictions': predictions,
        'targets': targets,
        'gene_names': gene_names,
        'extractor': extractor,
        'train_losses': train_losses
    }

# Usage Example
if __name__ == "__main__":
    # Run the complete pipeline without validation set
    results = run_dcrnn_gene_expression_pipeline(
        csv_file='/Users/beyzakaya/Desktop/timeSeries_HiC/mapped/mRNA/enhanced_interactions_synthetic_simple_mRNA.csv',
        seq_len=3,
        pred_len=1,
        hidden_dim=64,
        num_layers=2,
        batch_size=1,
        num_epochs=20,
        test_split=0.2
    )


"""
============================================================
FINAL RESULTS
============================================================
Mean Pearson Correlation: 0.4343
Mean Spearman Correlation: 0.3533
Mean RMSE: 5.1591
Valid genes evaluated: 50/50

Top 5 genes (Pearson): ['Igfbp3', 'PRIM2', 'MGAT4A', 'VIM', 'INMT']
Top 5 genes (Spearman): ['Tnc', 'INMT', 'Igfbp3', 'Lrp2', 'MGAT4A']
Bottom 5 genes (Pearson): ['tbp', 'Vegf', 'Claudin5', 'F13A1', 'HPGDS']

"""