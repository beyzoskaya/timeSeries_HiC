import os
import numpy as np
import torch
from utils import calculate_correlation, process_batch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.spatial.distance import cdist
import seaborn as sns

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

    def get_predictions(sequences, labels):
        predictions = []
        targets = []
        model.eval()
        with torch.no_grad():
            for seq, label in zip(sequences, labels):
                pred = model(seq)
                predictions.append(pred.squeeze().cpu().numpy())
                targets.append(label[0].x.squeeze().cpu().numpy())
        return np.vstack(predictions), np.vstack(targets)
    
    train_pred, train_true = get_predictions(train_sequences, train_labels)
    val_pred, val_true = get_predictions(val_sequences, val_labels)
    
    gene_connections = {}
    for gene in dataset.node_map.keys():
        gene_idx = dataset.node_map[gene]
        connections = len([1 for edge in dataset.edge_index.t() 
                         if edge[0] == gene_idx or edge[1] == gene_idx]) // 2
        gene_connections[gene] = connections
    
    gene_metrics = {}
    for gene, gene_idx in dataset.node_map.items():
        train_corr, _ = pearsonr(train_pred[:, gene_idx], train_true[:, gene_idx])
        val_corr, _ = pearsonr(val_pred[:, gene_idx], val_true[:, gene_idx])
        gene_metrics[gene] = {
            'train_corr': train_corr,
            'val_corr': val_corr,
            'connections': gene_connections[gene]
        }
    
    plt.figure(figsize=(12, 8))
    x = [m['connections'] for m in gene_metrics.values()]
    y_train = [m['train_corr'] for m in gene_metrics.values()]
    y_val = [m['val_corr'] for m in gene_metrics.values()]
    
    plt.scatter(x, y_train, alpha=0.5, label='Training', c='blue')
    plt.scatter(x, y_val, alpha=0.5, label='Validation', c='red')
    
    problematic_genes = ['THTPA', 'AMACR', 'MMP7', 'ABCG2', 'HPGDS', 'VIM']
    for gene in problematic_genes:
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
    
    # Training vs Validation Correlation
    plt.figure(figsize=(12, 8))
    plt.scatter([m['train_corr'] for m in gene_metrics.values()],
                [m['val_corr'] for m in gene_metrics.values()],
                alpha=0.5)
    
    min_corr = min(min([m['train_corr'] for m in gene_metrics.values()]),
                   min([m['val_corr'] for m in gene_metrics.values()]))
    max_corr = max(max([m['train_corr'] for m in gene_metrics.values()]),
                   max([m['val_corr'] for m in gene_metrics.values()]))
    plt.plot([min_corr, max_corr], [min_corr, max_corr], 'r--', alpha=0.5)
    
    for gene in problematic_genes:
        plt.annotate(gene, 
                    (gene_metrics[gene]['train_corr'], gene_metrics[gene]['val_corr']),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Training Correlation')
    plt.ylabel('Validation Correlation')
    plt.title('Training vs Validation Correlations')
    plt.grid(True, alpha=0.3)
    plt.savefig('plottings_STGCN/train_vs_val_correlation.png')
    plt.close()
    
    #  Prediction vs True Value Plot for Problematic Genes
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Prediction vs True Values for Problematic Genes')
    
    for idx, gene in enumerate(problematic_genes):
        row = idx // 3
        col = idx % 3
        gene_idx = dataset.node_map[gene]
        
        axs[row, col].scatter(val_true[:, gene_idx], val_pred[:, gene_idx], alpha=0.5)
        
        min_val = min(val_true[:, gene_idx].min(), val_pred[:, gene_idx].min())
        max_val = max(val_true[:, gene_idx].max(), val_pred[:, gene_idx].max())
        axs[row, col].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        
        axs[row, col].set_title(f'{gene}\nCorr: {gene_metrics[gene]["val_corr"]:.3f}')
        axs[row, col].grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig('plottings_STGCN/problematic_genes_predictions.png')
    plt.close()
    
    plt.figure(figsize=(12, 6))
    sns.histplot(data=list(gene_connections.values()), bins=20)
    plt.xlabel('Number of Connections')
    plt.ylabel('Count')
    plt.title('Distribution of Gene Connections')
    plt.savefig('plottings_STGCN/connection_distribution.png')
    plt.close()
    
    return gene_metrics