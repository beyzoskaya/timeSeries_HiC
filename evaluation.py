import os
import numpy as np
import torch
from utils import calculate_correlation, process_batch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.spatial.distance import cdist
import seaborn as sns
import pandas as pd
from create_graph_and_embeddings_STGCN import clean_gene_name

def analyze_temporal_patterns(model, val_sequences, val_labels, dataset, save_dir = 'temporal_analysis'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    print("\n=== Temporal Pattern Analysis ===")
    with torch.no_grad():
        predictions_over_time = []
        targets_over_time = []

        for seq,label in zip(val_sequences, val_labels):
            x, target = process_batch(seq, label)
            output = model(x)

            # Take only the last timestep
            target = target[:, :, -1:, :].cpu().numpy()
            output = output[:, :, -1:, :].cpu().numpy()

            predictions_over_time.append(output)
            targets_over_time.append(target)

        predictions = np.concatenate(predictions_over_time, axis=0)
        targets = np.concatenate(targets_over_time, axis=0)

        plt.figure(figsize=(15,10))
        genes = list(dataset.node_map.keys())
        num_genes_to_plot = min(6, len(genes))

        for i in range(num_genes_to_plot):
            plt.subplot(3,2, i+1)
            gene_idx = dataset.node_map[genes[i]]
            # Plot actual vs predicted values over time
            plt.plot(targets[:, 0, 0, gene_idx], 'b-', label='Actual', alpha=0.7)
            plt.plot(predictions[:, 0, 0, gene_idx], 'r--', label='Predicted', alpha=0.7)
            
            plt.title(f'Gene: {genes[i]}')
            plt.xlabel('Time Steps')
            plt.ylabel('Expression Level')
            plt.legend()

            actual = targets[:, 0, 0, gene_idx]
            print(f"Actual: {actual}")
            pred = predictions[:, 0, 0, gene_idx]
            print(f"Pred: {pred}")

            # Calculate rate of change
            actual_changes = np.diff(actual)
            pred_changes = np.diff(pred)
            # Direction accuracy
            direction_accuracy = np.mean(np.sign(actual_changes) == np.sign(pred_changes))
            
            print(f"\nTemporal Analysis for {genes[i]}:")
            print(f"Direction Accuracy: {direction_accuracy:.4f}")
            print(f"Max Change (Actual): {np.max(np.abs(actual_changes)):.4f}")
            print(f"Max Change (Predicted): {np.max(np.abs(pred_changes)):.4f}")
            
            # Check for pattern repetition
            actual_autocorr = np.correlate(actual, actual, mode='full')
            pred_autocorr = np.correlate(pred, pred, mode='full')
            
            print(f"Pattern Periodicity Match: {np.corrcoef(actual_autocorr, pred_autocorr)[0,1]:.4f}")
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/temporal_patterns.png')
        plt.close()

        print("\n=== Overall Temporal Statistics ===")
        
        # Rate of change distribution
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(np.diff(targets).flatten(), bins=50, alpha=0.5, label='Actual')
        plt.hist(np.diff(predictions).flatten(), bins=50, alpha=0.5, label='Predicted')
        plt.title('Distribution of Expression Changes')
        plt.xlabel('Change in Expression')
        plt.ylabel('Frequency')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.scatter(np.diff(targets).flatten(), np.diff(predictions).flatten(), alpha=0.1)
        plt.xlabel('Actual Changes')
        plt.ylabel('Predicted Changes')
        plt.title('Change Prediction Accuracy')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/temporal_changes.png')
        plt.close()


def analyze_interactions(model, val_sequences, val_labels, dataset):
    """Analyzes the preservation of gene-gene interactions."""
    model.eval()
    all_predicted = []
    all_true = []
    
    with torch.no_grad():
        for seq, label in zip(val_sequences, val_labels):
            x, target = process_batch(seq, label)  # Shape: [1, 32, seq_len, 52]
            output = model(x)  # Shape: [1, 32, seq_len, 52]
            
            # Extract predicted interactions (correlation between genes)
            predicted_corr = calculate_correlation(output)  # [channels, channels]
            true_corr = calculate_correlation(target)  # [channels, channels]
            
            # Collect results for interaction loss
            all_predicted.append(predicted_corr.cpu().numpy())
            all_true.append(true_corr.cpu().numpy())
    
    predicted_interactions = np.concatenate(all_predicted, axis=0)
    true_interactions = np.concatenate(all_true, axis=0)
    
    interaction_loss = np.mean((predicted_interactions - true_interactions) ** 2)
    
    print(f"Interaction Preservation (MSE between predicted and true interaction matrix): {interaction_loss:.4f}")
    
    return interaction_loss

def analyze_gene(model, val_sequences, val_labels, dataset):
    """Analyzes the gene prediction performance over time for validation sequences."""
    model.eval()
    all_predicted = []
    all_true = []
    
    with torch.no_grad():
        for seq, label in zip(val_sequences, val_labels):
            x, target = process_batch(seq, label)  # Shape: [1, 32, seq_len, 52]
            output = model(x)  # Shape: [1, 32, seq_len, 52]
            
            # remove the batch dimension, keeping seq_len and gene dimensions
            all_predicted.append(output.squeeze(0).cpu().numpy())  # Shape: [seq_len, genes]
            all_true.append(target.squeeze(0).cpu().numpy())  # Shape: [seq_len, genes]
    
    all_predicted = np.concatenate(all_predicted, axis=0)  # Shape: [total_time_steps, genes]
    all_true = np.concatenate(all_true, axis=0)  # Shape: [total_time_steps, genes]

    if all_predicted.shape[1] != all_true.shape[1]:
        # Align shapes by duplicating the true labels across all predicted genes
        if all_true.shape[1] == 1 and all_predicted.shape[1] > 1:
            all_true = np.repeat(all_true, all_predicted.shape[1], axis=1) 
        else:
            raise ValueError(f"Predicted and true arrays have different gene dimensions: "
                             f"{all_predicted.shape[1]} vs {all_true.shape[1]}")
    
    gene_corrs = []
    num_genes = all_true.shape[1]  
    
    for gene_idx in range(num_genes):  # Iterate over genes
        # Flatten the arrays to 1D
        pred_gene = all_predicted[:, gene_idx].flatten()  # Flatten to 1D
        true_gene = all_true[:, gene_idx].flatten()  # Flatten to 1D
        
        corr, _ = pearsonr(pred_gene, true_gene)
        gene_corrs.append(corr)
    
    mean_corr = np.mean(gene_corrs)
    
    print(f"Mean Pearson Correlation between predicted and true gene expressions: {mean_corr:.4f}")
    
    return mean_corr

def analyze_gene_characteristics(dataset, high_corr_genes, low_corr_genes):
    hic_stats = {}
    expr_stats = {}
    
    for gene in high_corr_genes + low_corr_genes:
        gene_rows = dataset.df[
            (dataset.df['Gene1_clean'] == gene) | 
            (dataset.df['Gene2_clean'] == gene)
        ]
        
        if len(gene_rows) > 0:
            if gene in low_corr_genes:
                hic_values = np.log1p(gene_rows['HiC_Interaction'].values)
            else:
                hic_values = gene_rows['HiC_Interaction'].values

            gene1_comps = gene_rows[gene_rows['Gene1_clean'] == gene]['Gene1_Compartment']
            gene2_comps = gene_rows[gene_rows['Gene2_clean'] == gene]['Gene2_Compartment']
            
            compartment = gene1_comps.iloc[0] if len(gene1_comps) > 0 else gene2_comps.iloc[0]
            
            hic_stats[gene] = {
                'mean_hic': np.mean(hic_values),
                'std_hic': np.std(hic_values),
                'max_hic': np.max(hic_values),
                'compartment': compartment
            }
        else:
            print(f"Warning: No HiC data found for gene {gene}")
            hic_stats[gene] = {
                'mean_hic': 0,
                'std_hic': 0,
                'max_hic': 0,
                'compartment': 'Unknown'
            }

        expressions = []
        for t in dataset.time_points:
            gene1_expr = dataset.df[dataset.df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
            gene2_expr = dataset.df[dataset.df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
            
            if len(gene1_expr) > 0:
                expressions.append(gene1_expr[0])
            elif len(gene2_expr) > 0:
                expressions.append(gene2_expr[0])
        
        if expressions:
            expr_stats[gene] = {
                'expr_mean': np.mean(expressions),
                'expr_std': np.std(expressions),
                'expr_range': max(expressions) - min(expressions),
                'expr_values': expressions
            }
        else:
            print(f"Warning: No expression data found for gene {gene}")
            expr_stats[gene] = {
                'expr_mean': 0,
                'expr_std': 0,
                'expr_range': 0,
                'expr_values': []
            }
    
    return hic_stats, expr_stats

def print_gene_analysis(dataset):
    high_corr_genes = ['VIM', 'INMT', 'Tnc', 'ADAMTSL2', 'Shisa3', 'FGF18']
    low_corr_genes = ['AMACR', 'ABCG2', 'MMP7', 'HPGDS', 'MGAT4A']
    
    print("\nAnalyzing gene characteristics...")
    hic_stats, expr_stats = analyze_gene_characteristics(dataset, high_corr_genes, low_corr_genes)
    
    # Print detailed statistics for each group
    print("\n=== High Correlation Genes Analysis ===")
    print("=" * 50)
    for gene in high_corr_genes:
        print(f"\n{gene}:")
        print("-" * 30)
        print("HiC Statistics:")
        print(f"  Mean: {hic_stats[gene]['mean_hic']:.2f}")
        print(f"  Std:  {hic_stats[gene]['std_hic']:.2f}")
        print(f"  Max:  {hic_stats[gene]['max_hic']:.2f}")
        print(f"  Compartment: {hic_stats[gene]['compartment']}")
        
        print("Expression Statistics:")
        print(f"  Mean: {expr_stats[gene]['expr_mean']:.2f}")
        print(f"  Std:  {expr_stats[gene]['expr_std']:.2f}")
        print(f"  Range: {expr_stats[gene]['expr_range']:.2f}")
    
    print("\n=== Low Correlation Genes Analysis ===")
    print("=" * 50)
    for gene in low_corr_genes:
        print(f"\n{gene}:")
        print("-" * 30)
        print("HiC Statistics:")
        print(f"  Mean: {hic_stats[gene]['mean_hic']:.2f}")
        print(f"  Std:  {hic_stats[gene]['std_hic']:.2f}")
        print(f"  Max:  {hic_stats[gene]['max_hic']:.2f}")
        print(f"  Compartment: {hic_stats[gene]['compartment']}")
        
        print("Expression Statistics:")
        print(f"  Mean: {expr_stats[gene]['expr_mean']:.2f}")
        print(f"  Std:  {expr_stats[gene]['expr_std']:.2f}")
        print(f"  Range: {expr_stats[gene]['expr_range']:.2f}")
    
    # Compare group statistics
    print("\n=== Group Comparisons ===")
    print("=" * 50)
    
    # HiC comparisons
    high_mean_hic = np.mean([hic_stats[g]['mean_hic'] for g in high_corr_genes])
    low_mean_hic = np.mean([hic_stats[g]['mean_hic'] for g in low_corr_genes])
    
    high_std_hic = np.mean([hic_stats[g]['std_hic'] for g in high_corr_genes])
    low_std_hic = np.mean([hic_stats[g]['std_hic'] for g in low_corr_genes])
    
    print("\nHiC Comparison:")
    print(f"High correlation genes average HiC: {high_mean_hic:.2f} ± {high_std_hic:.2f}")
    print(f"Low correlation genes average HiC:  {low_mean_hic:.2f} ± {low_std_hic:.2f}")
    
    # Expression comparisons
    high_mean_expr = np.mean([expr_stats[g]['expr_mean'] for g in high_corr_genes])
    low_mean_expr = np.mean([expr_stats[g]['expr_mean'] for g in low_corr_genes])
    
    high_std_expr = np.mean([expr_stats[g]['expr_std'] for g in high_corr_genes])
    low_std_expr = np.mean([expr_stats[g]['expr_std'] for g in low_corr_genes])
    
    print("\nExpression Comparison:")
    print(f"High correlation genes average expression: {high_mean_expr:.2f} ± {high_std_expr:.2f}")
    print(f"Low correlation genes average expression:  {low_mean_expr:.2f} ± {low_std_expr:.2f}")
    
    # Compartment analysis
    high_comps = [hic_stats[g]['compartment'] for g in high_corr_genes]
    low_comps = [hic_stats[g]['compartment'] for g in low_corr_genes]
    
    print("\nCompartment Distribution:")
    print("High correlation genes:")
    for comp in set(high_comps):
        count = high_comps.count(comp)
        print(f"  Compartment {comp}: {count} genes ({count/len(high_comps)*100:.1f}%)")
    
    print("Low correlation genes:")
    for comp in set(low_comps):
        count = low_comps.count(comp)
        print(f"  Compartment {comp}: {count} genes ({count/len(low_comps)*100:.1f}%)")
    
    return hic_stats, expr_stats

def calculate_temporal_metrics(true_values, predicted_values):

    metrics = {}
    
    # 1. Dynamic Time Warping (DTW) distance
    dtw_distances = []
    for i in range(true_values.shape[1]):
        true_seq = true_values[:, i].reshape(-1, 1)
        pred_seq = predicted_values[:, i].reshape(-1, 1)
        D = cdist(true_seq, pred_seq)
        dtw_distances.append(D.sum())
    metrics['dtw_mean'] = np.mean(dtw_distances)
    
    # 2. Temporal Correlation (considers lag relationships)
    def temporal_corr(y_true, y_pred, max_lag=3):
        correlations = []
        for lag in range(max_lag + 1):
            if lag == 0:
                corr = np.corrcoef(y_true, y_pred)[0, 1]
                #print(f"I'm inside lag=0")
            else:
                corr = np.corrcoef(y_true[lag:], y_pred[:-lag])[0, 1]
            correlations.append(corr)
        return np.max(correlations)  # Return max correlation across lags
    
    temp_corrs = []
    for i in range(true_values.shape[1]):
        temp_corr = temporal_corr(true_values[:, i], predicted_values[:, i])
        temp_corrs.append(temp_corr)
    metrics['temporal_correlation'] = np.mean(temp_corrs)
    
    # 3. Trend Consistency (direction of changes)
    def trend_accuracy(y_true, y_pred):
        true_diff = np.diff(y_true)
        pred_diff = np.diff(y_pred)
        true_direction = np.sign(true_diff)
        pred_direction = np.sign(pred_diff)
        return np.mean(true_direction == pred_direction)
    
    trend_accs = []
    for i in range(true_values.shape[1]):
        trend_acc = trend_accuracy(true_values[:, i], predicted_values[:, i])
        trend_accs.append(trend_acc)
    metrics['trend_accuracy'] = np.mean(trend_accs)
    
    # 4. RMSE over time
    time_rmse = []
    for t in range(true_values.shape[0]):
        rmse = np.sqrt(np.mean((true_values[t] - predicted_values[t])**2))
        time_rmse.append(rmse)
    metrics['rmse_over_time'] = np.mean(time_rmse)
    metrics['rmse_std'] = np.std(time_rmse)
    
    # 5. Temporal Pattern Similarity
    def pattern_similarity(y_true, y_pred):
        # Normalize sequences
        y_true_norm = y_true
        y_pred_norm = y_pred
        # Calculate similarity
        return np.mean(np.abs(y_true_norm - y_pred_norm))
    
    pattern_sims = []
    for i in range(true_values.shape[1]):
        sim = pattern_similarity(true_values[:, i], predicted_values[:, i])
        pattern_sims.append(sim)
    metrics['pattern_similarity'] = np.mean(pattern_sims)
    
    return metrics

def analyze_prediction_changes(predictions, targets, dataset):
    """Analyze how well the model predicts changes in expression."""
    actual_changes = np.diff(targets, axis=0)
    pred_changes = np.diff(predictions, axis=0)
    
    # Calculate direction accuracy
    direction_accuracy = np.mean(np.sign(actual_changes) == np.sign(pred_changes))
    
    # Calculate magnitude accuracy
    magnitude_ratio = np.abs(pred_changes) / (np.abs(actual_changes) + 1e-8)
    magnitude_accuracy = np.mean((magnitude_ratio >= 0.5) & (magnitude_ratio <= 2.0))
    
    print("\nChange Prediction Analysis:")
    print(f"Direction Accuracy: {direction_accuracy:.4f}")
    print(f"Magnitude Accuracy: {magnitude_accuracy:.4f}")
    print(f"Average Actual Change: {np.abs(actual_changes).mean():.4f}")
    print(f"Average Predicted Change: {np.abs(pred_changes).mean():.4f}")

def create_detailed_gene_plots(predictions, targets, dataset, save_dir):
    save_dir='plottings_STGCN'
    os.makedirs(save_dir, exist_ok=True)
    genes = list(dataset.node_map.keys())
    num_genes = len(genes)

    genes_per_page = 5
    num_pages = (num_genes + genes_per_page - 1) // genes_per_page

    for page in range(num_pages):
        start_idx = page * genes_per_page
        end_idx = min((page + 1) * genes_per_page, num_genes)
        current_genes = genes[start_idx:end_idx]

        fig, axes = plt.subplots(5, 4, figsize=(10, 20))
        fig.suptitle(f'Gene Predictions vs Actuals (Page {page+1}/{num_pages})', fontsize=16)
        axes = axes.flatten()

        for i,gene in enumerate(current_genes):
            gene_idx = dataset.node_map[gene]
            pred = predictions[:,0,gene_idx]
            true = targets[:,0, gene_idx]

            if not (np.isnan(pred).any() or np.isnan(true).any()):
                corr, p_value = pearsonr(pred, true)
                corr_text = f'Correlation: {corr:.3f}\np-value: {p_value:.3e}'
            else:
                corr_text = 'Correlation: N/A'

            # Create subplot
            ax = axes[i]
            ax.plot(true, label='Actual', marker='o')
            ax.plot(pred, label='Predicted', marker='s')
            ax.set_title(f'Gene: {gene}\n{corr_text}')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Expression Level')
            ax.legend()
            ax.grid(True)

            rmse = np.sqrt(np.mean((pred - true) ** 2))
            ax.text(0.05, 0.95, f'RMSE: {rmse:.3f}', 
                   transform=ax.transAxes, 
                   verticalalignment='top')
        
        for j in range(len(current_genes), len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/gene_predictions_page_{page+1}.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()


def create_gene_analysis_plots(model, train_sequences, train_labels, val_sequences, val_labels, dataset):
    
    def get_predictions_for_plotting(model, sequences, labels):
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for seq, label in zip(sequences, labels):
                x, target = process_batch(seq, label)
                output = model(x)
                
                # Take last time step
                output = output[:, :, -1:, :]
                
                # Convert to numpy and reshape
                pred = output.squeeze().cpu().numpy()
                true = target.squeeze().cpu().numpy()
                
                if len(pred.shape) == 1:
                    pred = pred.reshape(1, -1)
                if len(true.shape) == 1:
                    true = true.reshape(1, -1)
                
                all_predictions.append(pred)
                all_targets.append(true)

        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        return predictions, targets
    
    def plot_connected_genes_expression(dataset, target_gene):
        original_connections = set()
        original_df = pd.read_csv('mapped/enhanced_interactions_new_new.csv')
        original_df['Gene1_clean'] = original_df['Gene1'].apply(clean_gene_name)
        original_df['Gene2_clean'] = original_df['Gene2'].apply(clean_gene_name)

        for _, row in original_df.iterrows():
            if row['Gene1_clean'] == target_gene:
                original_connections.add(row['Gene2_clean'])
            elif row['Gene2_clean'] == target_gene:
                original_connections.add(row['Gene1_clean'])
                
        print(f"Original connections: {original_connections}")
        
        all_connections = set()
        for _, row in dataset.df.iterrows():
            if row['Gene1_clean'] == target_gene:
                all_connections.add(row['Gene2_clean'])
            elif row['Gene2_clean'] == target_gene:
                all_connections.add(row['Gene1_clean'])
                
        synthetic_connections = all_connections - original_connections
        print(f"Synthetic connections: {synthetic_connections}")
        
        plt.figure(figsize=(10, 6))
        
        expressions = get_gene_expressions(dataset, target_gene)
        plt.plot(dataset.time_points, expressions, label=target_gene, linewidth=2)
        
        for gene in original_connections:
            expressions = get_gene_expressions(dataset, gene)
            plt.plot(dataset.time_points, expressions, label=f"{gene} (original)", linewidth=2)
            
        for gene in synthetic_connections:
            expressions = get_gene_expressions(dataset, gene)
            plt.plot(dataset.time_points, expressions, label=f"{gene} (synthetic)", 
                    linewidth=1.5, linestyle='--')
        
        plt.xlabel('Time Points')
        plt.ylabel('Expression Value')
        plt.title(f'Expression Profiles for {target_gene}: Original vs Synthetic Connections')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'plottings_STGCN/{target_gene}_connected_genes.png')
        plt.close()

    def get_gene_expressions(dataset, gene):
        expressions = []
        for t in dataset.time_points:
            gene1_expr = dataset.df[dataset.df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
            gene2_expr = dataset.df[dataset.df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
            expr_value = gene1_expr[0] if len(gene1_expr) > 0 else gene2_expr[0] if len(gene2_expr) > 0 else np.nan
            expressions.append(expr_value)
        return expressions
    
    print("Getting training predictions...")
    train_pred, train_true = get_predictions_for_plotting(model, train_sequences, train_labels)
    print(f"Training predictions shape: {train_pred.shape}")
    
    print("Getting validation predictions...")
    val_pred, val_true = get_predictions_for_plotting(model, val_sequences, val_labels)
    print(f"Validation predictions shape: {val_pred.shape}")

    print("Calculating gene connections...")
    gene_connections = {}
    for gene in dataset.node_map.keys():
        gene_idx = dataset.node_map[gene]
        connections = len([1 for edge in dataset.edge_index.t() 
                         if edge[0] == gene_idx or edge[1] == gene_idx]) // 2
        gene_connections[gene] = connections
    
    print("Calculating gene metrics...")
    gene_metrics = {}
    for gene, gene_idx in dataset.node_map.items():
        try:
            train_corr, _ = pearsonr(train_pred[:, gene_idx], train_true[:, gene_idx])
            val_corr, _ = pearsonr(val_pred[:, gene_idx], val_true[:, gene_idx])
            
            gene_metrics[gene] = {
                'train_corr': train_corr,
                'val_corr': val_corr,
                'connections': gene_connections[gene],
                'train_rmse': np.sqrt(mean_squared_error(train_true[:, gene_idx], train_pred[:, gene_idx])),
                'val_rmse': np.sqrt(mean_squared_error(val_true[:, gene_idx], val_pred[:, gene_idx]))
            }
        except Exception as e:
            print(f"Error calculating metrics for gene {gene}: {str(e)}")
            continue
    
    print("Creating plots...")
    
    # 1. Correlation vs Connections Scatter Plot
    plt.figure(figsize=(12, 8))
    x = [m['connections'] for m in gene_metrics.values()]
    y_train = [m['train_corr'] for m in gene_metrics.values()]
    y_val = [m['val_corr'] for m in gene_metrics.values()]
    
    plt.scatter(x, y_train, alpha=0.5, label='Training', c='blue')
    plt.scatter(x, y_val, alpha=0.5, label='Validation', c='red')
    
    # Add labels for problematic genes
    problematic_genes = ['THTPA', 'AMACR', 'MMP7', 'ABCG2', 'HPGDS', 'VIM']
    for gene in problematic_genes:
        if gene in gene_metrics:
            plt.annotate(gene, 
                        (gene_metrics[gene]['connections'], gene_metrics[gene]['val_corr']),
                        xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Number of Connections')
    plt.ylabel('Correlation')
    plt.title('Gene Correlations vs Number of Connections')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('plottings_STGCN/correlation_vs_connections.png')
    plt.close()
    
    # 2. Training vs Validation Correlation
    plt.figure(figsize=(12, 8))
    plt.scatter([m['train_corr'] for m in gene_metrics.values()],
                [m['val_corr'] for m in gene_metrics.values()],
                alpha=0.5)
    
    # Add diagonal line
    min_corr = min(min([m['train_corr'] for m in gene_metrics.values()]),
                   min([m['val_corr'] for m in gene_metrics.values()]))
    max_corr = max(max([m['train_corr'] for m in gene_metrics.values()]),
                   max([m['val_corr'] for m in gene_metrics.values()]))
    plt.plot([min_corr, max_corr], [min_corr, max_corr], 'r--', alpha=0.5)
    
    # Add labels for problematic genes
    for gene in problematic_genes:
        if gene in gene_metrics:
            plt.annotate(gene, 
                        (gene_metrics[gene]['train_corr'], gene_metrics[gene]['val_corr']),
                        xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Training Correlation')
    plt.ylabel('Validation Correlation')
    plt.title('Training vs Validation Correlations')
    plt.grid(True, alpha=0.3)
    plt.savefig('plottings_STGCN/train_vs_val_correlation.png')
    plt.close()
    
    # 3. Prediction vs True Value Plot for Problematic Genes
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Prediction vs True Values for Problematic Genes')
    
    for idx, gene in enumerate(problematic_genes):
        if gene in gene_metrics:
            row = idx // 3
            col = idx % 3
            gene_idx = dataset.node_map[gene]
            
            axs[row, col].scatter(val_true[:, gene_idx], val_pred[:, gene_idx], alpha=0.5)
 
            min_val = min(val_true[:, gene_idx].min(), val_pred[:, gene_idx].min())
            max_val = max(val_true[:, gene_idx].max(), val_pred[:, gene_idx].max())
            axs[row, col].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
            
            axs[row, col].set_title(f'{gene}\nCorr: {gene_metrics[gene]["val_corr"]:.3f}\nRMSE: {gene_metrics[gene]["val_rmse"]:.3f}')
            axs[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plottings_STGCN/problematic_genes_predictions.png')
    plt.close()
    
    # 4. Connection Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data=list(gene_connections.values()), bins=20)
    plt.xlabel('Number of Connections')
    plt.ylabel('Count')
    plt.title('Distribution of Gene Connections')
    plt.savefig('plottings_STGCN/connection_distribution.png')
    plt.close()

    neg_genes = ['HPGDS', 'ABCG2', 'AMACR']
    for gene in neg_genes:
        expressions = []
        for t in dataset.time_points:
            #print(f"Length of time points: {len(dataset.time_points)} ")
            gene1_expr = dataset.df[dataset.df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
            gene2_expr = dataset.df[dataset.df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
            expr_value = gene1_expr[0] if len(gene1_expr) > 0 else \
                        (gene2_expr[0] if len(gene2_expr) > 0 else 0.0)
            expressions.append(expr_value)
        
        plt.plot(dataset.time_points, expressions, label=gene)

    plt.xlabel('Time Points')
    plt.ylabel('Expression Value')
    plt.title('Expression Profiles of Negatively Correlated Genes')
    plt.legend()
    #plt.show()
    plt.savefig('plottings_STGCN/expr_values_negative_corel_genes')
    plt.close()

    best_genes = ['VIM', 'INMT', 'Tnc', 'ADAMTSL2', 'Shisa3']
    for gene in best_genes:
        expressions = []
        for t in dataset.time_points:
            gene1_expr = dataset.df[dataset.df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
            gene2_expr = dataset.df[dataset.df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
            expr_value = gene1_expr[0] if len(gene1_expr) > 0 else \
                        (gene2_expr[0] if len(gene2_expr) > 0 else 0.0)
            expressions.append(expr_value)
        
        plt.plot(dataset.time_points, expressions, label=gene)

    plt.xlabel('Time Points')
    plt.ylabel('Expression Value')
    plt.title('Expression Profiles of Best Correlated Genes')
    plt.legend()
    #plt.show()
    plt.savefig('plottings_STGCN/expr_values_best_corel_genes')
    plt.close()

    plot_connected_genes_expression(dataset, 'AMACR') # E2F8 and TGFB1 genes have same expression values for the AMACR
    plot_connected_genes_expression(dataset, 'ABCG2')
    plot_connected_genes_expression(dataset, 'HPGDS')

    print("\nSummary Statistics:")
    print(f"Mean Training Correlation: {np.mean([m['train_corr'] for m in gene_metrics.values()]):.4f}")
    print(f"Mean Validation Correlation: {np.mean([m['val_corr'] for m in gene_metrics.values()]):.4f}")
    print("\nProblematic Genes Statistics:")
    for gene in problematic_genes:
        if gene in gene_metrics:
            print(f"\n{gene}:")
            print(f"Training Correlation: {gene_metrics[gene]['train_corr']:.4f}")
            print(f"Validation Correlation: {gene_metrics[gene]['val_corr']:.4f}")
            print(f"Number of Connections: {gene_metrics[gene]['connections']}")
            print(f"Training RMSE: {gene_metrics[gene]['train_rmse']:.4f}")
            print(f"Validation RMSE: {gene_metrics[gene]['val_rmse']:.4f}")
    
    return gene_metrics