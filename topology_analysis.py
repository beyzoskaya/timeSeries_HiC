import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import requests
import seaborn as sns
import json

def get_string_interactions(gene_list, species=10090, score_threshold=700):
    """
    species=10090 for mouse
    score_threshold=700 means 0.7 confidence (STRING uses 0-1000 scale)
    """
    string_api_url = "https://string-db.org/api"
    output_format = "json"
    method = "network"

    params = {
        "identifiers": "\r".join(gene_list),    # your protein list
        "species": species,                      # species NCBI identifier 
        "required_score": score_threshold,       # confidence score
        "network_flavor": "confidence",          # show confidence links
        "caller_identity": "www.awesome_app.org" # your app name
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
        'Density': nx.density(G),
        'Avg Clustering': nx.average_clustering(G),
        'Avg Degree': np.mean(list(dict(G.degree()).values())),
        'Transitivity': nx.transitivity(G),
        'Avg Path Length': nx.average_shortest_path_length(G) if nx.is_connected(G) else 0
    }
    
    categories = list(metrics.keys())
    n = len(categories)
    
    angles = [n/float(n) * 2 * np.pi * i for i in range(n)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    values = list(metrics.values())
    values += values[:1]
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    plt.title('Network Metrics Overview')
    
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
    
    base_fig.savefig('network_visualization.png')
    dist_fig.savefig('centrality_distributions.png')
    subnet_fig.savefig('subnetworks.png')
    heatmap_fig.savefig('interaction_heatmap.png')
    radar_fig.savefig('network_metrics_radar.png')
    
    print("Visualizations have been saved as:")
    print("1. network_visualization.png - Basic network layout")
    print("2. centrality_distributions.png - Distribution of network metrics")
    print("3. subnetworks.png - Top subnetworks/communities")
    print("4. interaction_heatmap.png - Interaction strength heatmap")
    print("5. network_metrics_radar.png - Overview of network metrics")
    
    return G, metrics, centrality_df, [base_fig, dist_fig, subnet_fig, heatmap_fig, radar_fig]


if __name__ == "__main__":
    genes = [
     "Hist1h1b", "VIM", "P-63", "INMT", "ADAMTSL2", "Tnc", "FGF18", "Shisa3", "integrin subunit alpha 8", "Hist1h2ab", 
     "CD38", "MMP-3", "Lrp2", "ppia", "THTPA", "Vegf", "GATA-6", "ABCA3", "Kcnma1", "tfrc", "RAGE", "F13A1", "MCPt4",
     "FOXF2", "EPHA7", "AGER", "hmbs", "E2F8", "TGFB1", "TTF-1", "Claudin5", "GUCY1A2  sGC", "PRIM2", "tbp", "SFTP-D",
     "N-Cadherin", "Thy1", "Claudin 1", "Igfbp3", "EGFR", "ywhaz", "hprt", "ABCD1", "NME3", "MGAT4A", "MMP7", "HPGDS",
     "ABCG2", "AMACR"
    ]
    G, metrics, centrality_df, figures = main_with_visualizations(genes)
    plt.show()

