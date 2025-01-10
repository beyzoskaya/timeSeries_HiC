from create_graph_and_embeddings_STGCN import *
from STGCN_training import  train_stgcn
from STGCN_training import * 
import json
from evaluation import *
from utils import process_batch, calculate_correlation

def filter_genes(dataset, gene_list):
        """Return a subset of the dataset containing only the specified genes."""
        filtered_node_map = {gene: idx for idx, gene in enumerate(gene_list)}
        filtered_dataset = TemporalGraphDataset(
        csv_file='mapped/enhanced_interactions.csv',
        embedding_dim=32,
        seq_len=3,
        pred_len=1
    )
        filtered_dataset.node_map = filtered_node_map
        filtered_dataset.base_graph = {gene: dataset.base_graph[gene] for gene in gene_list}
        filtered_dataset.df = dataset.df[dataset.df['Gene1_clean'].isin(gene_list) | dataset.df['Gene2_clean'].isin(gene_list)]
        return filtered_dataset

def train_cluster_models(dataset, clusters, cluster_blocks):
    """Train separate models for each cluster."""
    cluster_models = {}
    cluster_metrics = {}

    for cluster_name, genes in clusters.items():
        print(f"\nTraining model for {cluster_name} with {len(genes)} genes.")
        
        # Create filtered dataset for the current cluster
        cluster_dataset = filter_genes(dataset, genes)
        
        # Create Args with cluster-specific blocks
        args = Args(custom_blocks=cluster_blocks[cluster_name])
        
        # Train model for this cluster
        model, val_sequences, val_labels, train_losses, val_losses = train_stgcn(cluster_dataset, val_ratio=0.2)
        
        # Evaluate model
        metrics = evaluate_model_performance(model, val_sequences, val_labels, cluster_dataset)
        
        cluster_models[cluster_name] = model
        cluster_metrics[cluster_name] = metrics

        # Print the metrics as per the format you want
        print(f"Completed training for {cluster_name} cluster.")
        print(f"Metrics: {metrics['Overall']}")
        
        # Printing the performance summary as you requested:
        print("\nModel Performance Summary:")
        
        # Overall Metrics
        print("\nOverall Metrics:")
        for metric, value in metrics['Overall'].items():
            print(f"{metric}: {value:.4f}")

        # Gene Performance
        print("\nGene Performance:")
        print(f"Mean Gene Correlation: {metrics['Gene']['Mean_Correlation']:.4f}")
        print(f"Best Performing Genes: {', '.join(metrics['Gene']['Best_Genes'])}")

        # Temporal Performance
        print("\nTemporal Performance:")
        print(f"Time-lagged Correlation: {metrics['Temporal']['Mean_Temporal_Correlation']:.4f}")
        print(f"DTW Distance: {metrics['Temporal']['Mean_DTW_Distance']:.4f}")
        print(f"Direction Accuracy: {metrics['Temporal']['Mean_Direction_Accuracy']:.4f}")
        print(f"Change Magnitude Ratio: {metrics['Temporal']['Change_Magnitude_Ratio']:.4f}")

        
        print(f"Completed training for {cluster_name} cluster.")
        print(f"Metrics: {metrics['Overall']}")

    return cluster_models, cluster_metrics


class Args:
    def __init__(self, custom_blocks=None):
        self.Kt = 3  # temporal kernel size
        self.Ks = 3  # spatial kernel size
        self.n_his = 3  # historical sequence length
        self.n_pred = 1
        self.blocks = custom_blocks if custom_blocks else [
            [32, 32, 32],    
            [32, 16, 16],    
            [16, 8, 1]
        ]
        self.act_func = 'glu'
        self.graph_conv_type = 'graph_conv'
        self.enable_bias = True
        self.droprate = 0

def main():
    dataset = TemporalGraphDataset(
        csv_file='mapped/enhanced_interactions.csv',
        embedding_dim=32,
        seq_len=3,
        pred_len=1
    )

    clusters, gene_expressions = analyze_expression_levels(dataset)
    
    cluster_blocks = {
        'high_expr': [
            [32, 48, 48],    # Wider for high expression genes
            [48, 32, 32],
            [32, 16, 1]
        ],
        'medium_expr': [
            [32, 32, 32],    # Standard for medium expression
            [32, 24, 24],
            [24, 16, 1]
        ],
        'low_expr': [
            [32, 24, 24],    
            [24, 16, 16],
            [16, 8, 1]
        ]
    }

    cluster_models, cluster_metrics = train_cluster_models(dataset, clusters, cluster_blocks)

    torch.save(cluster_models, 'cluster_models.pth')
    
    print("\nCluster-based training completed. Models and metrics saved.")

if __name__ == "__main__":
    main()