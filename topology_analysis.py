import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import requests
import seaborn as sns
import json
from scipy import stats

def get_string_interactions(gene_list, species=10090, score_threshold=700):
    """
    species=10090 for mouse
    score_threshold=700 means 0.7 confidence (STRING uses 0-1000 scale)
    """
    string_api_url = "https://string-db.org/api"
    output_format = "json"
    method = "network"

    params = {
        "identifiers": "\r".join(gene_list),  
        "species": species,                     
        "required_score": score_threshold,       
        "network_flavor": "confidence",          
        "caller_identity": "www.awesome_app.org" 
    }

    response = requests.post(f"{string_api_url}/{output_format}/{method}", data=params)
    
    if response.status_code != 200:
        print(f"Error accessing STRING database: {response.status_code}")
        return None
        
    return response.json()

def create_network_from_string(string_data):
    if not string_data:
        return None
        
    G = nx.Graph()
    
    for interaction in string_data:
        nodeA = interaction.get('preferredName_A', interaction.get('stringId_A'))
        nodeB = interaction.get('preferredName_B', interaction.get('stringId_B'))
        score = float(interaction.get('score', 0)) / 1000  # Convert to 0-1 scale
        
        G.add_edge(nodeA, nodeB, weight=score)
    
    return G

def visualize_network(G, centrality_measure='degree'):
    if G is None or G.number_of_nodes() == 0:
        print("No network to visualize")
        return None
        
    plt.figure(figsize=(12, 8))
    
    if centrality_measure == 'degree':
        centrality = nx.degree_centrality(G)
    elif centrality_measure == 'betweenness':
        centrality = nx.betweenness_centrality(G)
    else:
        try:
            centrality = nx.eigenvector_centrality(G, max_iter=1000)
        except:
            centrality = nx.degree_centrality(G)
            print("Falling back to degree centrality")
    
    pos = nx.spring_layout(G, k=1/np.sqrt(G.number_of_nodes()))
    
    nx.draw_networkx_nodes(G, pos, 
                          node_size=[v * 5000 for v in centrality.values()],
                          node_color=list(centrality.values()),
                          cmap=plt.cm.viridis)
    
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, alpha=edge_weights)

    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title(f"PPI Network (size/color: {centrality_measure} centrality)")
    plt.axis('off')
    
    return plt.gcf()

def identify_hub_genes(G, top_n=5):
    if G is None or G.number_of_nodes() == 0:
        return None, None
        
    degree_cent = nx.degree_centrality(G)
    between_cent = nx.betweenness_centrality(G)
    
    try:
        eigen_cent = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        eigen_cent = {node: 0 for node in G.nodes()}
        print("Could not compute eigenvector centrality")
    
    cent_df = pd.DataFrame({
        'Degree_Centrality': degree_cent,
        'Betweenness_Centrality': between_cent,
        'Eigenvector_Centrality': eigen_cent
    })
    
    hub_genes = {
        'degree_hubs': sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:top_n],
        'betweenness_hubs': sorted(between_cent.items(), key=lambda x: x[1], reverse=True)[:top_n],
        'eigenvector_hubs': sorted(eigen_cent.items(), key=lambda x: x[1], reverse=True)[:top_n]
    }
    
    return hub_genes, cent_df

def analyze_network_metrics(G):

    if G is None or G.number_of_nodes() == 0:
        return None
        
    metrics = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': nx.density(G),
        'average_clustering': nx.average_clustering(G),
        'degree_centrality': nx.degree_centrality(G),
        'betweenness_centrality': nx.betweenness_centrality(G)
    }
    
    try:
        metrics['eigenvector_centrality'] = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        print("Could not compute eigenvector centrality")
        metrics['eigenvector_centrality'] = None
    
    return metrics

def main(gene_list, species=10090, score_threshold=700):

    print("Fetching protein interactions from STRING...")
    string_data = get_string_interactions(gene_list, species, score_threshold)
    
    if string_data is None:
        print("Could not fetch data from STRING")
        return None, None, None, None
    
    print("Creating network...")
    G = create_network_from_string(string_data)
    
    if G is None or G.number_of_nodes() == 0:
        print("No network could be created with the given parameters")
        return None, None, None, None
    
    print("Analyzing network metrics...")
    metrics = analyze_network_metrics(G)
    
    print("Identifying hub genes...")
    hub_genes, centrality_df = identify_hub_genes(G)
    
    print("Creating visualization...")
    fig = visualize_network(G)
    
    print("\nNetwork Summary:")
    print(f"Number of nodes: {metrics['nodes']}")
    print(f"Number of edges: {metrics['edges']}")
    print(f"Network density: {metrics['density']:.3f}")
    print(f"Average clustering coefficient: {metrics['average_clustering']:.3f}")
    
    print("\nTop Hub Genes (by degree centrality):")
    for gene, score in hub_genes['degree_hubs']:
        print(f"{gene}: {score:.3f}")
    
    return G, metrics, centrality_df, fig

def visualize_centrality_distribution(G):
    """
    Degree centrality measures how many direct connections a gene has. 
    Higher values indicate hub genes that interact with many others. --> Most genes have low degree centrality, but a few have high values.

    Betweenness centrality reflects how often a gene acts as a bridge between other genes in the network. --> Most genes have very low betweenness, while a few have much higher values.

    The clustering coefficient measures how well a gene's neighbors are connected to each other. --> Many genes have either very low clustering or very high clustering.
    """
    degree_cent = pd.Series(nx.degree_centrality(G))
    between_cent = pd.Series(nx.betweenness_centrality(G))
    clustering_coef = pd.Series(nx.clustering(G))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    sns.histplot(degree_cent, kde=True, ax=axes[0])
    axes[0].set_title('Degree Centrality Distribution')
    
    sns.histplot(between_cent, kde=True, ax=axes[1])
    axes[1].set_title('Betweenness Centrality Distribution')
    
    sns.histplot(clustering_coef, kde=True, ax=axes[2])
    axes[2].set_title('Clustering Coefficient Distribution')
    
    plt.tight_layout()
    return fig

def visualize_subnetworks(G, top_n=3):
    communities = nx.community.greedy_modularity_communities(G)
    
    fig, axes = plt.subplots(1, min(top_n, len(communities)), figsize=(15, 5))
    if len(communities) == 1:
        axes = [axes]
    
    for i, community in enumerate(list(communities)[:top_n]):
        subgraph = G.subgraph(community)
        pos = nx.spring_layout(subgraph)
        
        axes[i].set_title(f'Subnetwork {i+1}\n({len(community)} nodes)')
        nx.draw_networkx(subgraph, pos=pos, ax=axes[i],
                        node_color='lightblue',
                        node_size=500,
                        font_size=8,
                        width=2)
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def visualize_interaction_heatmap(G):
    """
    Represents the interaction strength between different genes. 
    Darker red regions indicate stronger interactions, while lighter or white regions indicate weaker or no interactions.
    """
    adj_matrix = nx.to_pandas_adjacency(G)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(adj_matrix, 
                cmap='YlOrRd',
                xticklabels=True, 
                yticklabels=True)
    plt.title('Gene Interaction Heatmap')
    return plt.gcf()

def plot_network_metrics_radar(G):

    metrics = {
        'Network\nDensity': nx.density(G),
        'Average\nClustering': nx.average_clustering(G),
        'Average\nDegree': np.mean(list(dict(G.degree()).values())),
        'Network\nTransitivity': nx.transitivity(G),
        'Average\nPath Length': nx.average_shortest_path_length(G) if nx.is_connected(G) else 0
    }
    
    categories = list(metrics.keys())
    n = len(categories)
   
    angles = [n/float(n) * 2 * np.pi * i for i in range(n)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    values = list(metrics.values())
    values += values[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, color='blue')
    ax.fill(angles, values, alpha=0.25, color='blue')
    
    ax.set_xticklabels([])
    
    for idx, (angle, label) in enumerate(zip(angles[:-1], categories)):
        label_position = 1.3  
        
        x = label_position * np.cos(angle)
        y = label_position * np.sin(angle)
        
        ax.text(angle, label_position, label,
                ha='center', va='center',
                size=12, fontweight='bold',
                bbox=dict(facecolor='white', 
                         edgecolor='none',
                         alpha=0.7,
                         pad=2))
        
        value = values[idx]
        value_position = value * 0.7  
        ax.text(angle, value_position, f'{value:.3f}',
                ha='center', va='center',
                size=10,
                bbox=dict(facecolor='white',
                         edgecolor='none',
                         alpha=0.7,
                         pad=1))
    
    plt.title('Network Metrics Overview', size=16, pad=30, y=1.1)
    
    ax.grid(True, linestyle='--', alpha=0.3)
    
    ax.set_rlim(0, 1.5)
    
    return fig

def main_with_visualizations(gene_list, species=10090, score_threshold=700):
    G, metrics, centrality_df, base_fig = main(gene_list, species, score_threshold)
    
    if G is None:
        return
    
    print("\nGenerating additional visualizations...")
    dist_fig = visualize_centrality_distribution(G)
    
    subnet_fig = visualize_subnetworks(G)
    
    heatmap_fig = visualize_interaction_heatmap(G)
    
    radar_fig = plot_network_metrics_radar(G)
    
    base_fig.savefig('plottings_STGCN/ppi_analysis/network_visualization.png')
    dist_fig.savefig('plottings_STGCN/ppi_analysis/centrality_distributions.png')
    subnet_fig.savefig('plottings_STGCN/ppi_analysis/subnetworks.png')
    heatmap_fig.savefig('plottings_STGCN/ppi_analysis/interaction_heatmap.png')
    radar_fig.savefig('plottings_STGCN/ppi_analysis/network_metrics_radar.png')
    
    print("Visualizations have been saved as:")
    print("1. network_visualization.png - Basic network layout")
    print("2. centrality_distributions.png - Distribution of network metrics")
    print("3. subnetworks.png - Top subnetworks/communities")
    print("4. interaction_heatmap.png - Interaction strength heatmap")
    print("5. network_metrics_radar.png - Overview of network metrics")
    
    return G, metrics, centrality_df, [base_fig, dist_fig, subnet_fig, heatmap_fig, radar_fig]

def perform_statistical_analysis(G, correlation_data, threshold=0.8):
    
    print("Genes in network:", list(G.nodes()))
    print("Genes in correlation data:", list(correlation_data.keys()))
 
    network_genes = set(G.nodes())
    correlation_genes = set(correlation_data.keys())

    missing_genes = correlation_genes - network_genes
    if missing_genes:
        print("\nWarning: The following genes are in correlation data but not in network:")
        print(missing_genes)
    
    valid_genes = network_genes.intersection(correlation_genes)
    print(f"\nNumber of valid genes for analysis: {len(valid_genes)}")
    
    high_perf = {k: v for k, v in correlation_data.items() 
                if v['correlation'] >= threshold and k in valid_genes}
    low_perf = {k: v for k, v in correlation_data.items() 
                if v['correlation'] < threshold and k in valid_genes}
    
    try:
        high_metrics = {
            'connections': [v['connections'] for v in high_perf.values()],
            'centrality': [nx.degree_centrality(G)[gene] for gene in high_perf.keys()],
            'clustering': [nx.clustering(G)[gene] for gene in high_perf.keys()]
        }
        
        low_metrics = {
            'connections': [v['connections'] for v in low_perf.values()],
            'centrality': [nx.degree_centrality(G)[gene] for gene in low_perf.keys()],
            'clustering': [nx.clustering(G)[gene] for gene in low_perf.keys()]
        }
    except KeyError as e:
        print(f"Error accessing gene: {e}")
        return None
    
    results = {}
    if high_metrics['connections'] and low_metrics['connections']:
        for metric in ['connections', 'centrality', 'clustering']:
            try:
                statistic, pvalue = stats.mannwhitneyu(
                    high_metrics[metric], 
                    low_metrics[metric],
                    alternative='two-sided'
                )
                results[f'{metric}_test'] = {
                    'statistic': statistic,
                    'p_value': pvalue,
                    'high_mean': np.mean(high_metrics[metric]),
                    'low_mean': np.mean(low_metrics[metric])
                }
            except Exception as e:
                print(f"Error in statistical test for {metric}: {e}")
                results[f'{metric}_test'] = None
    
    return results

def visualize_correlation_patterns(correlation_data):

    correlations = [v['correlation'] for v in correlation_data.values()]
    connections = [v['connections'] for v in correlation_data.values()]
    genes = list(correlation_data.keys())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.histplot(correlations, kde=True, ax=ax1)
    ax1.set_title('Distribution of Prediction Correlations')
    ax1.set_xlabel('Correlation')
    ax1.set_ylabel('Count')
    
    mean_corr = np.mean(correlations)
    median_corr = np.median(correlations)
    ax1.axvline(mean_corr, color='r', linestyle='--', label=f'Mean: {mean_corr:.3f}')
    ax1.axvline(median_corr, color='g', linestyle='--', label=f'Median: {median_corr:.3f}')
    ax1.legend()
    
    sns.regplot(x=connections, y=correlations, ax=ax2)
    ax2.set_title('Correlation vs Number of Connections')
    ax2.set_xlabel('Number of Connections')
    ax2.set_ylabel('Correlation')
    
    corr_coef = np.corrcoef(connections, correlations)[0,1]
    ax2.text(0.05, 0.95, f'Correlation: {corr_coef:.3f}', 
             transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def identify_significant_patterns(G, correlation_data, threshold=0.8):
    results = {}
    
    degree_cent = nx.degree_centrality(G)
    between_cent = nx.betweenness_centrality(G)
    clustering_coef = nx.clustering(G)

    high_perf = {k: v['correlation'] for k, v in correlation_data.items() 
                if v['correlation'] >= threshold}
    low_perf = {k: v['correlation'] for k, v in correlation_data.items() 
                if v['correlation'] < threshold}
    
    high_perf_hubs = {gene: degree_cent[gene] for gene in high_perf 
                      if degree_cent[gene] > np.mean(list(degree_cent.values()))}
    low_perf_hubs = {gene: degree_cent[gene] for gene in low_perf 
                     if degree_cent[gene] > np.mean(list(degree_cent.values()))}
    
    results['high_performing_hubs'] = high_perf_hubs
    results['low_performing_hubs'] = low_perf_hubs
    
    results['metrics_comparison'] = {
        'high_perf': {
            'avg_degree': np.mean([degree_cent[gene] for gene in high_perf]),
            'avg_betweenness': np.mean([between_cent[gene] for gene in high_perf]),
            'avg_clustering': np.mean([clustering_coef[gene] for gene in high_perf])
        },
        'low_perf': {
            'avg_degree': np.mean([degree_cent[gene] for gene in low_perf]),
            'avg_betweenness': np.mean([between_cent[gene] for gene in low_perf]),
            'avg_clustering': np.mean([clustering_coef[gene] for gene in low_perf])
        }
    }
    
    return results


def enhanced_main(gene_list, correlation_data, species=10090, score_threshold=700):
    G, metrics, centrality_df, figures = main_with_visualizations(gene_list)

    statistical_results = perform_statistical_analysis(G, correlation_data)
    correlation_patterns_fig = visualize_correlation_patterns(correlation_data)
    significant_patterns = identify_significant_patterns(G, correlation_data)
    
    print("\nStatistical Analysis Results:")
    for metric, result in statistical_results.items():
        print(f"\n{metric}:")
        print(f"p-value: {result['p_value']:.4f}")
        print(f"High performing mean: {result['high_mean']:.4f}")
        print(f"Low performing mean: {result['low_mean']:.4f}")
    
    print("\nSignificant Hub Genes:")
    print("High performing hubs:", list(significant_patterns['high_performing_hubs'].keys()))
    print("Low performing hubs:", list(significant_patterns['low_performing_hubs'].keys()))
    
    return G, metrics, statistical_results, significant_patterns

def analyze_network_importance(G, correlation_data=None, top_n=5):
    """
    Top Hub Genes: EGFR, TGFB1, VEGFA, CDH2, FGF18
    These genes have the most direct interactions.
    Key regulators in biological pathways.
    May control multiple processes like signaling, proliferation, differentiation.

    Top Bottleneck Genes: EGFR, TFRC, YWHAZ, VEGFA, HMBS
    These genes connect different gene clusters.
    They act as "communication highways" in the network.
    Disrupting these genes could severely impact network function.
    These genes might be critical for linking distinct pathways (e.g., growth signaling, metabolism, stress response).

    Top Essential Genes: EGFR, TGFB1, FGF18, VEGFA, CDH2
    These genes have high influence over the network.
    They interact with other highly connected genes.
    Removing them would significantly disrupt the network.
    These genes are likely core regulators of biological function.
    They may control large gene expression programs.
    High eigenvector centrality suggests they are master regulators of pathways.


    """
    degree_cent = nx.degree_centrality(G)
    between_cent = nx.betweenness_centrality(G)
    eigen_cent = nx.eigenvector_centrality(G, max_iter=1000)
    
    plt.figure(figsize=(15, 10))
    
    node_sizes = [v * 3000 for v in degree_cent.values()]
    
    node_colors = list(between_cent.values())

    pos = nx.spring_layout(G, k=1/np.sqrt(G.number_of_nodes()))

    nx.draw_networkx_nodes(G, pos, 
                          node_size=node_sizes,
                          node_color=node_colors,
                          cmap=plt.cm.viridis)
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title("Gene Interaction Network\nNode size: Degree centrality, Color: Betweenness centrality")
    
    print("\nImportant Genes in Network:")
    
    print(f"\nTop {top_n} Hub Genes (Degree Centrality):")
    hub_genes = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:top_n]
    for gene, score in hub_genes:
        print(f"- {gene}: {score:.3f}")
    
    print(f"\nTop {top_n} Bottleneck Genes (Betweenness Centrality):")
    bottleneck_genes = sorted(between_cent.items(), key=lambda x: x[1], reverse=True)[:top_n]
    for gene, score in bottleneck_genes:
        print(f"- {gene}: {score:.3f}")
    
    print(f"\nTop {top_n} Essential Genes (Eigenvector Centrality):")
    essential_genes = sorted(eigen_cent.items(), key=lambda x: x[1], reverse=True)[:top_n]
    for gene, score in essential_genes:
        print(f"- {gene}: {score:.3f}")
    
    if correlation_data is not None:
        print("\nCorrelation values for important genes:")
        important_genes = set([g[0] for g in hub_genes + bottleneck_genes + essential_genes])
        for gene in important_genes:
            if gene in correlation_data:
                corr = correlation_data[gene]['correlation']
                print(f"- {gene}: correlation = {corr:.3f}")
    
    plt.tight_layout()
    return plt.gcf()

        
if __name__ == "__main__":
    gene_list = [
     "Hist1h1b", "VIM", "P-63", "INMT", "ADAMTSL2", "Tnc", "FGF18", "Shisa3", "integrin subunit alpha 8", "Hist1h2ab", 
     "CD38", "MMP-3", "Lrp2", "ppia", "THTPA", "Vegf", "GATA-6", "ABCA3", "Kcnma1", "tfrc", "RAGE", "F13A1", "MCPt4",
     "FOXF2", "EPHA7", "AGER", "hmbs", "E2F8", "TGFB1", "TTF-1", "Claudin5", "GUCY1A2  sGC", "PRIM2", "tbp", "SFTP-D",
     "N-Cadherin", "Thy1", "Claudin 1", "Igfbp3", "EGFR", "ywhaz", "hprt", "ABCD1", "NME3", "MGAT4A", "MMP7", "HPGDS",
     "ABCG2", "AMACR"
    ]

    correlation_data = {
    'Hist1h1b': {'correlation': 0.9734, 'connections': 3},
    'VIM': {'correlation': 0.9682, 'connections': 7},
    'P-63': {'correlation': 0.9653, 'connections': 3},
    'INMT': {'correlation': 0.9509, 'connections': 2},
    'ADAMTSL2': {'correlation': 0.9423, 'connections': 5},
    'Tnc': {'correlation': 0.9390, 'connections': 1},
    'FGF18': {'correlation': 0.9342, 'connections': 3},
    'Shisa3':{ 'correlation': 0.9239, 'connections': 1},
    'integrin subunit alpha 8': {'correlation': 0.9040, 'connections': 6},
    'Hist1h2ab': {'correlation': 0.9039, 'connections': 3},
    'CD38': {'correlation': 0.9004, 'connections': 1},
    'MMP-3': {'correlation': 0.8857, 'connections': 4},
    'Lrp2': {'correlation': 0.8754, 'connections': 4},
    'ppia': {'correlation': 0.8724, 'connections': 4},
    'THTPA': {'correlation': 0.8696, 'connections': 6},
    'Vegf': {'correlation': 0.8454, 'connections': 6},
    'GATA-6': {'correlation': 0.8435, 'connections': 2},
    'ABCA3': {'correlation': 0.8393, 'connections': 8},
    'Kcnma1': {'correlation': 0.8368, 'connections': 4},
    'tfrc': {'correlation': 0.8356, 'connections': 6},
    'RAGE': {'correlation': 0.8014, 'connections': 5},
    'F13A1': {'correlation': 0.8011, 'connections': 3},
    'MCPt4': {'correlation': 0.7904, 'connections': 7},
    'FOXF2': {'correlation': 0.7877, 'connections': 3},
    'EPHA7': {'correlation': 0.7525, 'connections': 2},
    'AGER': {'correlation': 0.7421,'connections': 5},
    'hmbs': {'correlation': 0.7192, 'connections': 4},
    'E2F8': {'correlation': 0.7061, 'connections': 4},
    'TGFB1': {'correlation': 0.7021, 'connections': 4},
    'TTF-1': {'correlation': 0.6943, 'connections': 5},
    'Claudin5': {'correlation': 0.6479, 'connections': 3},
    'GUCY1A2  sGC': {'correlation': 0.6435, 'connections': 4},
    'PRIM2': {'correlation': 0.6426, 'connections': 4},
    'tbp': {'correlation': 0.6201, 'connections': 6},
    'SFTP-D': {'correlation': 0.6044, 'connections': 3},
    'N-Cadherin': {'correlation': 0.6003, 'connections': 1},
    'Thy1': {'correlation': 0.5764, 'connections': 4},
    'Claudin 1': {'correlation': 0.5182, 'connections': 3},
    'Igfbp3': {'correlation': 0.5062, 'connections': 4},
    'EGFR': {'correlation': 0.5011, 'connections': 6},
    'ywhaz': {'correlation': 0.4336,'connections': 3},
    'hprt': {'correlation': 0.4130, 'connections': 4},
    'ABCD1': {'correlation': 0.3923, 'connections': 2},
    'NME3': {'correlation': 0.3050, 'connections': 7},
    'MGAT4A': {'correlation': 0.2948, 'connections': 5},
    'MMP7': {'correlation': 0.1937, 'connections': 4},
    'HPGDS': {'correlation': 0.1415, 'connections': 6},
    'ABCG2': {'correlation': -0.2441, 'connections': 3},
    'AMACR': {'correlation': -0.4294, 'connections': 3}
}

    #G, metrics, stats_results, patterns = enhanced_main(gene_list, correlation_data)
    #plt.show()

    G, metrics, centrality_df, figures = main_with_visualizations(gene_list)
    fig = analyze_network_importance(G, correlation_data)
    plt.show()
    
