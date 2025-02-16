import pandas as pd
import gseapy as gp
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('/Users/beyzakaya/Desktop/temporal gene/mapped/enhanced_interactions_synthetic_simple.csv')
expression_columns = [col for col in df.columns if 'Time' in col]

unique_genes = pd.concat([df['Gene1'], df['Gene2']]).unique()
unique_genes = [str(gene).upper() for gene in unique_genes]

expression_matrix = pd.DataFrame(index=unique_genes)

for time_col in expression_columns:
    gene1_values = df.set_index('Gene1')[time_col]
    gene2_values = df.set_index('Gene2')[time_col]
    
    combined_values = pd.concat([gene1_values, gene2_values])
    mean_values = combined_values.groupby(combined_values.index.str.upper()).mean()
    
    expression_matrix[time_col] = mean_values

expression_matrix = expression_matrix.fillna(0)

early_time_points = [col for col in expression_columns if float(col.split('_')[-1]) <= 13.0]
late_time_points = [col for col in expression_columns if float(col.split('_')[-1]) > 13.0]

cls = ['Early'] * len(early_time_points) + ['Late'] * len(late_time_points)

gene_list = [
    "HIST1H1B", "VIM", "P-63", "INMT", "ADAMTSL2", "TNC", "FGF18", "SHISA3", 
    "INTEGRIN SUBUNIT ALPHA 8", "HIST1H2AB", "CD38", "MMP-3", "LRP2", "PPIA", 
    "THTPA", "VEGF", "GATA-6", "ABCA3", "KCNMA1", "TFRC", "RAGE", "F13A1", "MCPt4",
    "FOXF2", "EPHA7", "AGER", "HMBS", "E2F8", "TGFB1", "TTF-1", "CLAUDIN5", "GUCY1A2  SGC", 
    "PRIM2", "TBP", "SFTP-D", "N-CADHERIN", "THY1", "CLAUDIN 1", "IGFBP3", "EGFR", "YWHAZ", 
    "HPRT", "ABCD1", "NME3", "MGAT4A", "MMP7", "HPGDS", "ABCG2", "AMACR"
]
gene_list = [gene.upper() for gene in gene_list]

try:
    gsea = gp.gsea(
        data=expression_matrix[early_time_points + late_time_points],
        gene_sets={'my_gene_set': gene_list},
        cls=cls,
        method='signal_to_noise',
        permutation_type='phenotype',
        number_of_permutations=1000,
        verbose=True
    )
    
    gsea_results = gsea.res2d
    print("\nGSEA Results:")
    print(gsea_results)
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    result_metrics = pd.DataFrame({
        'Metric': ['Enrichment Score (ES)', 'Normalized ES', 'Nominal p-value', 'FDR q-value'],
        'Value': [
            gsea_results.loc[0, 'ES'],
            gsea_results.loc[0, 'NES'],
            gsea_results.loc[0, 'NOM p-val'],
            gsea_results.loc[0, 'FDR q-val']
        ]
    })
    
    sns.barplot(x='Metric', y='Value', data=result_metrics)
    plt.xticks(rotation=45)
    plt.title('GSEA Metrics')
    
    plt.subplot(2, 1, 2)
    leading_genes = gsea_results.loc[0, 'Lead_genes'].split(';')
    leading_gene_values = pd.DataFrame({
        'Gene': leading_genes,
        'Early_Mean': expression_matrix[early_time_points].loc[leading_genes].mean(axis=1),
        'Late_Mean': expression_matrix[late_time_points].loc[leading_genes].mean(axis=1)
    })
   
    leading_gene_matrix = leading_gene_values[['Early_Mean', 'Late_Mean']]
    sns.heatmap(leading_gene_matrix, 
                annot=True, 
                fmt='.2f',
                cmap='RdBu_r',
                yticklabels=leading_genes)
    plt.title('Expression Pattern of Leading Edge Genes')
    
    plt.tight_layout()
    plt.savefig('expr_patterns.png')
    plt.show()
    
    print("\nDetailed GSEA Analysis:")
    print(f"Enrichment Score (ES): {gsea_results.loc[0, 'ES']:.3f}")
    print(f"Normalized ES: {gsea_results.loc[0, 'NES']:.3f}")
    print(f"Nominal p-value: {gsea_results.loc[0, 'NOM p-val']:.3f}")
    print(f"FDR q-value: {gsea_results.loc[0, 'FDR q-val']:.3f}")
    print("\nLeading Edge Genes:")
    for gene in leading_genes:
        early_mean = expression_matrix[early_time_points].loc[gene].mean()
        late_mean = expression_matrix[late_time_points].loc[gene].mean()
        print(f"{gene}: Early mean = {early_mean:.2f}, Late mean = {late_mean:.2f}")
    
except Exception as e:
    print(f"Error during GSEA analysis: {str(e)}")
    print("\nShape of expression matrix:", expression_matrix.shape)
    print("Number of class labels:", len(cls))
    print("Sample of expression matrix:")
    print(expression_matrix.head())

def create_additional_gsea_visualizations(gsea, expression_matrix, early_time_points, late_time_points):
    gsea_results = gsea.res2d
    leading_genes = gsea_results.loc[0, 'Lead_genes'].split(';')
    
    # 1. Time Series Plot for Leading Edge Genes
    plt.figure(figsize=(15, 6))
    time_points = [float(col.split('_')[-1]) for col in early_time_points + late_time_points]
    all_timepoints = early_time_points + late_time_points
    
    for gene in leading_genes:
        gene_values = expression_matrix.loc[gene, all_timepoints].values
        plt.plot(time_points, gene_values, marker='o', label=gene)
    
    plt.axvline(x=4.0, color='r', linestyle='--', label='Early/Late Boundary')
    plt.xlabel('Time Points')
    plt.ylabel('Expression Values')
    plt.title('Expression Trajectories of Leading Edge Genes')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('expr_trajectories.png')
    
    # 2. Boxplot Comparison
    plt.figure(figsize=(12, 6))
    early_data = []
    late_data = []
    gene_labels = []
    
    for gene in leading_genes:
        early_values = expression_matrix.loc[gene, early_time_points].values
        late_values = expression_matrix.loc[gene, late_time_points].values
        early_data.extend(early_values)
        late_data.extend(late_values)
        gene_labels.extend([gene] * len(early_values))
        gene_labels.extend([gene] * len(late_values))
    
    comparison_df = pd.DataFrame({
        'Expression': early_data + late_data,
        'Time': ['Early'] * len(early_data) + ['Late'] * len(late_data),
        'Gene': gene_labels
    })
    
    sns.boxplot(x='Gene', y='Expression', hue='Time', data=comparison_df)
    plt.xticks(rotation=45)
    plt.title('Early vs Late Expression Distribution')
    plt.tight_layout()
    plt.savefig('early_late_distribution.png')
    
    # 3. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    gene_correlations = expression_matrix.loc[leading_genes, all_timepoints].T.corr()
    sns.heatmap(gene_correlations, 
                annot=True, 
                fmt='.2f', 
                cmap='coolwarm',
                center=0)
    plt.title('Correlation Between Leading Edge Genes')
    plt.tight_layout()
    plt.savefig('correl_between_edge_genes.png')
    
    # 4. Expression Change Plot
    plt.figure(figsize=(10, 6))
    early_means = expression_matrix.loc[leading_genes, early_time_points].mean(axis=1)
    late_means = expression_matrix.loc[leading_genes, late_time_points].mean(axis=1)
    fold_changes = late_means - early_means
    
    sns.barplot(x=leading_genes, y=fold_changes)
    plt.xticks(rotation=45)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Expression Changes (Late - Early)')
    plt.ylabel('Log2 Expression Change')
    plt.tight_layout()
    plt.savefig('expr_changes.png')
    
    return plt.gcf()

create_additional_gsea_visualizations(gsea, expression_matrix, early_time_points, late_time_points)
plt.show()
