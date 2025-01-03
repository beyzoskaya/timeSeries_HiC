import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

class GeneNormalizer:
    def __init__(self, feature_range=(0, 1)):
       
        self.feature_range = feature_range
        self.scalers = {}  # Dictionary to store scalers for each gene
        self.gene_stats = {}  # Store statistics for each gene
        
    def fit_transform_dataset(self, dataset):
        print("\n=== Gene Expression Statistics Before Normalization ===")
        
        # Get all time points and genes
        time_points = dataset.time_points
        genes = list(dataset.node_map.keys())
        
        # Create DataFrame for before statistics
        before_stats = []
        
        # Process each gene
        for gene in genes:
            gene_values = []
            
            # Collect all expression values for this gene across time points
            for t in time_points:
                gene1_expr = dataset.df[dataset.df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
                gene2_expr = dataset.df[dataset.df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
                
                values = np.concatenate([gene1_expr, gene2_expr])
                gene_values.extend(values[~np.isnan(values)])
            
            gene_values = np.array(gene_values)
            
            # Store original statistics
            stats = {
                'Gene': gene,
                'Original_Mean': np.mean(gene_values),
                'Original_Std': np.std(gene_values),
                'Original_Min': np.min(gene_values),
                'Original_Max': np.max(gene_values),
                'Original_Range': np.max(gene_values) - np.min(gene_values)
            }
            before_stats.append(stats)
            
            # Create and fit MinMaxScaler
            scaler = MinMaxScaler(feature_range=self.feature_range)
            gene_values_2d = gene_values.reshape(-1, 1)
            scaler.fit(gene_values_2d)
            self.scalers[gene] = scaler
            
            # Store statistics
            self.gene_stats[gene] = stats
        
        # Convert statistics to DataFrame and print summary
        stats_df = pd.DataFrame(before_stats)
        print("\nGene Expression Range Summary:")
        print(stats_df.describe())
        
        # Plot distribution of gene expressions before normalization
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        sns.boxplot(data=stats_df[['Original_Mean', 'Original_Std']])
        plt.title('Distribution of Gene Means and Standard Deviations')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(data=stats_df[['Original_Min', 'Original_Max', 'Original_Range']])
        plt.title('Distribution of Gene Ranges')
        
        plt.tight_layout()
        plt.savefig('gene_expression_distributions_before.png')
        plt.close()
        
        # Now normalize the data in the dataset
        for t in time_points:
            for gene in genes:
                # Update Gene1 columns
                mask1 = dataset.df['Gene1_clean'] == gene
                if mask1.any():
                    values = dataset.df.loc[mask1, f'Gene1_Time_{t}'].values.reshape(-1, 1)
                    dataset.df.loc[mask1, f'Gene1_Time_{t}'] = self.scalers[gene].transform(values).flatten()
                
                # Update Gene2 columns
                mask2 = dataset.df['Gene2_clean'] == gene
                if mask2.any():
                    values = dataset.df.loc[mask2, f'Gene2_Time_{t}'].values.reshape(-1, 1)
                    dataset.df.loc[mask2, f'Gene2_Time_{t}'] = self.scalers[gene].transform(values).flatten()
        
        # Collect and print statistics after normalization
        after_stats = []
        for gene in genes:
            gene_values = []
            for t in time_points:
                gene1_expr = dataset.df[dataset.df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
                gene2_expr = dataset.df[dataset.df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
                values = np.concatenate([gene1_expr, gene2_expr])
                gene_values.extend(values[~np.isnan(values)])
            
            gene_values = np.array(gene_values)
            after_stats.append({
                'Gene': gene,
                'Normalized_Mean': np.mean(gene_values),
                'Normalized_Std': np.std(gene_values),
                'Normalized_Min': np.min(gene_values),
                'Normalized_Max': np.max(gene_values),
                'Normalized_Range': np.max(gene_values) - np.min(gene_values)
            })
        
        after_stats_df = pd.DataFrame(after_stats)
        print("\n=== Gene Expression Statistics After Normalization ===")
        print(after_stats_df.describe())
        
        # Plot distribution after normalization
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        sns.boxplot(data=after_stats_df[['Normalized_Mean', 'Normalized_Std']])
        plt.title('Distribution of Normalized Gene Means and Standard Deviations')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(data=after_stats_df[['Normalized_Min', 'Normalized_Max', 'Normalized_Range']])
        plt.title('Distribution of Normalized Gene Ranges')
        
        plt.tight_layout()
        plt.savefig('gene_expression_distributions_after.png')
        plt.close()
        
        return stats_df, after_stats_df
    
    def inverse_transform(self, predictions, gene_indices):
        inversed_predictions = np.zeros_like(predictions)
        
        for idx, gene in gene_indices.items():
            if gene in self.scalers:
                scaler = self.scalers[gene]
                values = predictions[..., idx].reshape(-1, 1)
                inversed = scaler.inverse_transform(values)
                inversed_predictions[..., idx] = inversed.flatten()
        
        return inversed_predictions

def print_value_ranges(dataset, time_point=None):
    if time_point is None:
        time_points = dataset.time_points
    else:
        time_points = [time_point]
    
    all_values = []
    for t in time_points:
        # Get all Gene1 time columns
        gene1_values = dataset.df[f'Gene1_Time_{t}'].values
        # Get all Gene2 time columns
        gene2_values = dataset.df[f'Gene2_Time_{t}'].values
        
        all_values.extend(gene1_values)
        all_values.extend(gene2_values)
    
    all_values = np.array(all_values)
    print(f"\nValue Ranges:")
    print(f"Min: {np.min(all_values):.4f}")
    print(f"Max: {np.max(all_values):.4f}")
    print(f"Mean: {np.mean(all_values):.4f}")
    print(f"Std: {np.std(all_values):.4f}")