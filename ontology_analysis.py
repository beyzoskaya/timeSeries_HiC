import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def analyze_expression_levels(dataset):
    genes = list(dataset.node_map.keys())
    gene_expressions = {}

    for gene in genes:
        expressions = []
        for t in dataset.time_points:
            gene1_expr = dataset.df[dataset.df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
            gene2_expr = dataset.df[dataset.df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
            expr_values = np.concatenate([gene1_expr, gene2_expr])
            expressions.extend(expr_values)
        
        mean_expr = np.mean(expressions)
        gene_expressions[gene] = mean_expr
    
    expr_values = np.array(list(gene_expressions.values()))
    q1, q2, q3 = np.percentile(expr_values, [25, 50, 75])

    clusters = {
        'high_expr': [],
        'medium_expr': [],
        'low_expr': []
    }

    for gene, expr in gene_expressions.items():
        if expr > q3:
            clusters['high_expr'].append(gene)
        elif expr < q1:
            clusters['low_expr'].append(gene)
        else:
            clusters['medium_expr'].append(gene)
    
    print("\nExpression Cluster Analysis:")
    for cluster_name, genes in clusters.items():
        print(f"\n{cluster_name.upper()} Expression Cluster:")
        print(f"Number of genes: {len(genes)}")
        print(f"Average expression: {np.mean([gene_expressions[g] for g in genes]):.4f}")
        print("Genes:", ', '.join(genes[:5]), "..." if len(genes) > 5 else "")
    
    plt.figure(figsize=(10, 6))
    plt.hist(expr_values, bins=30, alpha=0.7)
    plt.axvline(q1, color='r', linestyle='--', label='Q1')
    plt.axvline(q2, color='g', linestyle='--', label='Q2')
    plt.axvline(q3, color='b', linestyle='--', label='Q3')
    plt.xlabel('Mean Expression Level')
    plt.ylabel('Number of Genes')
    plt.title('Distribution of Gene Expression Levels')
    plt.legend()
    plt.savefig('plottings_STGCN/expression_clusters.png')
    plt.close()

    return clusters, gene_expressions

def analyze_expression_levels_kmeans(dataset, n_clusters=3):
    genes = list(dataset.node_map.keys())
    gene_expressions = {}

    for gene in genes:
        expressions = []
        for t in dataset.time_points:
            # Extract expression values for Gene1_clean and Gene2_clean
            gene1_expr = dataset.df[dataset.df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
            gene2_expr = dataset.df[dataset.df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
            expr_values = np.concatenate([gene1_expr, gene2_expr])
            expressions.extend(expr_values)

        mean_expr = np.mean(expressions)
        gene_expressions[gene] = mean_expr
    
    # Prepare the feature matrix for KMeans (only mean expression values)
    mean_expr_values = np.array(list(gene_expressions.values())).reshape(-1, 1)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(mean_expr_values)

    # Map KMeans clusters to 'high_expr', 'medium_expr', 'low_expr' based on mean value order
    sorted_centers = np.argsort(kmeans.cluster_centers_.flatten())
    label_mapping = {sorted_centers[0]: 'low_expr', 
                      sorted_centers[1]: 'medium_expr', 
                      sorted_centers[2]: 'high_expr'}
    
    clusters = {
        'high_expr': [],
        'medium_expr': [],
        'low_expr': []
    }

    for gene, label in zip(genes, cluster_labels):
        cluster_name = label_mapping[label]
        clusters[cluster_name].append(gene)

    print("\nKMeans Expression Cluster Analysis:")
    for cluster_name, genes in clusters.items():
        mean_cluster_expr = np.mean([gene_expressions[g] for g in genes])
        print(f"\n{cluster_name.upper()} Expression Cluster:")
        print(f"Number of genes: {len(genes)}")
        print(f"Average expression: {mean_cluster_expr:.4f}")
        print("Genes:", ', '.join(genes[:5]), "..." if len(genes) > 5 else "")

    plt.figure(figsize=(10, 6))
    plt.hist(mean_expr_values, bins=30, alpha=0.7, label='Mean Expression Levels')
    for cluster_center in kmeans.cluster_centers_:
        plt.axvline(cluster_center, color='k', linestyle='--', label=f'Cluster Center: {cluster_center[0]:.4f}')
    plt.xlabel('Mean Expression Level')
    plt.ylabel('Number of Genes')
    plt.title('Distribution of Gene Expression Levels with KMeans Clusters')
    plt.legend()
    plt.savefig('plottings_STGCN/expression_clusters_kmeans.png')
    plt.close()

    return clusters, gene_expressions




