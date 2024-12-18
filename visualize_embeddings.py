import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import os

def visualize_embeddings(embedding_dir, time_point=-4.5):
    embeddings = {}
    filename = f"embeddings_time_{time_point}.txt"
    with open(f"{embedding_dir}/{filename}", 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            gene = parts[0]
            embedding = np.array([float(x) for x in parts[1:]])
            embeddings[gene] = embedding
    
    genes = list(embeddings.keys())
    X = np.array([embeddings[gene] for gene in genes])
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
    plt.title(f'PCA of Gene Embeddings at Time {time_point}')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    
    for i, gene in enumerate(genes):
        if i % max(1, len(genes)//20) == 0:  
            plt.annotate(gene, (X_pca[i, 0], X_pca[i, 1]))
    plt.savefig(f'{embedding_dir}/pca_visualization_time_{time_point}.png')
    plt.close()
  
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.5)
    plt.title(f't-SNE of Gene Embeddings at Time {time_point}')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')

    for i, gene in enumerate(genes):
        if i % max(1, len(genes)//20) == 0:
            plt.annotate(gene, (X_tsne[i, 0], X_tsne[i, 1]))
    plt.savefig(f'{embedding_dir}/tsne_visualization_time_{time_point}.png')
    plt.close()
    
    plt.figure(figsize=(12, 6))
    plt.hist(X.flatten(), bins=50)
    plt.title(f'Distribution of Embedding Values at Time {time_point}')
    plt.xlabel('Embedding Value')
    plt.ylabel('Frequency')
    plt.savefig(f'{embedding_dir}/embedding_distribution_time_{time_point}.png')
    plt.close()

    sample_size = min(20, len(genes))
    sample_genes = genes[:sample_size]
    sample_embeddings = X[:sample_size, :20]  
    
    plt.figure(figsize=(15, 8))
    sns.heatmap(sample_embeddings, xticklabels=range(20), yticklabels=sample_genes)
    plt.title(f'Heatmap of First 20 Embedding Dimensions at Time {time_point}')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Gene')
    plt.savefig(f'{embedding_dir}/embedding_heatmap_time_{time_point}.png')
    plt.close()

def visualize_temporal_patterns(embedding_dir, gene_name):
    time_files = [f for f in os.listdir(embedding_dir) if f.startswith('embeddings_time_')]
    temporal_data = {}
    
    for file in time_files:
        time = float(file.split('_')[-1].replace('.txt', ''))
        with open(f"{embedding_dir}/{file}", 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if parts[0] == gene_name:
                    temporal_data[time] = np.array([float(x) for x in parts[1:]])
    
    times = sorted(temporal_data.keys())
    embeddings = np.array([temporal_data[t] for t in times])

    plt.figure(figsize=(12, 6))
    for i in range(5):  
        plt.plot(times, embeddings[:, i], label=f'Dimension {i+1}')
    plt.title(f'Temporal Evolution of Embeddings for {gene_name}')
    plt.xlabel('Time')
    plt.ylabel('Embedding Value')
    plt.legend()
    plt.savefig(f'{embedding_dir}/temporal_evolution_{gene_name}.png')
    plt.close()
    
    plt.figure(figsize=(15, 8))
    sns.heatmap(embeddings[:, :20], xticklabels=range(20), yticklabels=times)
    plt.title(f'Temporal Changes in First 20 Dimensions for {gene_name}')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Time')
    plt.savefig(f'{embedding_dir}/temporal_heatmap_{gene_name}.png')
    plt.close()

if __name__ == "__main__":
    embedding_dir = "embeddings_txt"

    for time_point in [-4.5, 1.0, 14.0, 28.0]: 
        print(f"Visualizing embeddings for time {time_point}")
        visualize_embeddings(embedding_dir, time_point)
    
    sample_genes = ["ABCA3", "AGER"] 
    for gene in sample_genes:
        print(f"Visualizing temporal patterns for {gene}")
        visualize_temporal_patterns(embedding_dir, gene)