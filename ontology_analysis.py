import numpy as np
import matplotlib.pyplot as plt

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


