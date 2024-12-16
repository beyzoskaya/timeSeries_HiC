import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy import stats

def plot_data_insights(data_file):
    df = pd.read_csv(data_file)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df['HiC_Interaction'], bins=50, log=True)
    plt.title('Distribution of HiC Interaction Frequencies')
    plt.xlabel('Interaction Frequency (log scale)')
    plt.ylabel('Count')
    plt.savefig('plottings/hic_distribution.png')
    plt.close()
    
    time_cols = [col for col in df.columns if col.startswith('Gene1_Time_')]
    time_points = [float(col.split('_')[-1]) for col in time_cols]
    
    unique_genes = df['Gene1'].unique()
    sample_genes = np.random.choice(unique_genes, min(10, len(unique_genes)), replace=False)
    
    plt.figure(figsize=(15, 8))
    for gene in sample_genes:
        gene_data = df[df['Gene1'] == gene].iloc[0]
        expression_values = [gene_data[col] for col in time_cols]
        plt.plot(time_points, expression_values, label=gene, marker='o', alpha=0.7)
    
    plt.title('Gene Expression Time Series')
    plt.xlabel('Time Point')
    plt.ylabel('Expression Level')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('plottings/time_series.png')
    plt.close()
    
    plt.figure(figsize=(12, 12))
    G = nx.Graph()
    
    for _, row in df.iterrows():
        G.add_edge(row['Gene1'], row['Gene2'], 
                  weight=np.log1p(row['HiC_Interaction']))
    
    node_sizes = {}
    for gene in G.nodes():
        gene_rows = df[(df['Gene1'] == gene) | (df['Gene2'] == gene)]
        avg_expr = gene_rows[time_cols].mean().mean()
        node_sizes[gene] = avg_expr
    
    pos = nx.spring_layout(G, k=1/np.sqrt(len(G.nodes())), iterations=50)
    nx.draw_networkx_nodes(G, pos, 
                          node_size=[node_sizes[gene]*100 for gene in G.nodes()],
                          node_color='lightblue',
                          alpha=0.7)
    nx.draw_networkx_edges(G, pos, 
                          edge_color='gray',
                          width=[G[u][v]['weight']/10 for u,v in G.edges()],
                          alpha=0.5)
    plt.title('Gene Interaction Network\nNode size: Avg Expression, Edge width: HiC Interaction')
    plt.savefig('plottings/network.png')
    plt.close()
    

    plt.figure(figsize=(10, 6))
    distances = []
    correlations = []
    
    for _, row in df.iterrows():
        dist = abs(row['Gene1_Bin'] - row['Gene2_Bin'])
        expr1 = [row[col] for col in time_cols]
        expr2 = [row[col.replace('Gene1', 'Gene2')] for col in time_cols]
        corr = stats.pearsonr(expr1, expr2)[0]
        distances.append(dist)
        correlations.append(corr)
    
    plt.scatter(distances, correlations, alpha=0.5)
    plt.xlabel('Genomic Distance')
    plt.ylabel('Expression Correlation')
    plt.title('Expression Correlation vs Genomic Distance')
    plt.savefig('plottings/distance_correlation.png')
    plt.close()

plot_data_insights('mapped/enhanced_interactions.csv')