import os
os.environ["OMP_NUM_THREADS"] = '1'

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import torch
import argparse

import utils
import model
from scipy.stats import spearmanr, pearsonr
import numpy as np
import matplotlib.pyplot as plt
import random
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/gene_expression_over_time_mirna.csv')
    parser.add_argument('--output_path', default='clusters_mirna.txt')
    parser.add_argument('--n_layers_g', type=int, default=2)
    parser.add_argument('--n_layers_d', type=int, default=2)
    parser.add_argument('--n_layers_ta', type=int, default=2)
    parser.add_argument('--threshold_g', type=float, default=0.1)
    parser.add_argument('--threshold_d', type=float, default=0.1)
    parser.add_argument('--loss_t_weight', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    data = pd.read_csv(args.data_path, index_col=0)
    data = data.drop(columns=["Time_154.0"])
    n_genes = data.shape[0]
    print(f"Number of genes: {n_genes}")
    n_timestamps = data.shape[1]
    print(f"Number of timestamps: {n_timestamps}")
    filter_g = utils.create_filters(args.n_layers_g, n_timestamps)
    filter_d = utils.create_filters(args.n_layers_d, n_timestamps)

    x_d = []
    edges_d = []
    sil = -2

    print('***** Creating the graphs *****')

    x_g = torch.tensor(data.values, dtype=torch.float)
    adj = utils.correlations(data)
    edges_g, weights = utils.adj_to_edge(adj, args.threshold_g)

    for i in range(n_timestamps-1):
        x_d.append(x_g)
        gene_expression = data.iloc[:, i:i + 2]
        edges_d.append(utils.nearest_neighbors(gene_expression, args.threshold_d))

    # Initialize the model
    hagnet_model = model.HAGNET(n_genes, filter_g, filter_d, args.n_layers_ta, args.loss_t_weight)
    optimizer = torch.optim.Adam(hagnet_model.parameters(), lr=args.lr)

    print('***** Training the model *****')

    for i in range(args.epochs):
        hagnet_model.train()
        hagnet_model.zero_grad()
        output, loss = hagnet_model(x_g, edges_g, weights, x_d, edges_d)
        loss.backward()

        if ((i + 1) % 5 == 0):
            print(f'Epoch {i + 1}: Training Loss =', loss.item())
        output = output.detach().numpy()
        clusters_pred = utils.clustering(output)
        temp = utils.evaluation(data, clusters_pred)

        if temp > sil:
            sil = temp
            best_pred = clusters_pred

        optimizer.step()

    results = utils.clusters(best_pred)
    with open(args.output_path, "w") as file:
        for cluster, gene in results:
            file.write(f"Cluster_{cluster}\tGene_{gene}\n")


    print('***** Evaluating on test set *****')
    best_pred = np.array(best_pred)

    pearson_list = []
    spearman_list = []

    # For each gene (each row in data), compare predicted vs true expression profile
    for gene_idx in range(n_genes):
        # true expression over time
        true_values = data.iloc[gene_idx, :].values
        # predicted expression: take mean value of cluster the gene belongs to, per timestamp
        cluster_id = best_pred[gene_idx]

        # Compute predicted profile as mean of genes in the same cluster
        cluster_genes = np.where(best_pred == cluster_id)[0]
        cluster_profile = data.iloc[cluster_genes, :].mean(axis=0).values

        # compute Pearson and Spearman
        p_corr, _ = pearsonr(true_values, cluster_profile)
        s_corr, _ = spearmanr(true_values, cluster_profile)

        pearson_list.append(p_corr)
        spearman_list.append(s_corr)

    mean_pearson = np.nanmean(pearson_list)
    mean_spearman = np.nanmean(spearman_list)

    print(f"Mean Pearson correlation across genes: {mean_pearson:.4f}")
    print(f"Mean Spearman correlation across genes: {mean_spearman:.4f}")

    print('***** Plotting actual vs predicted expression profiles *****')

    plot_dir = 'gene_expression_plots_mirna'
    os.makedirs(plot_dir, exist_ok=True)

    timestamps = data.columns.values

    for gene_idx in range(n_genes):
        true_values = data.iloc[gene_idx, :].values
        cluster_id = best_pred[gene_idx]
        cluster_genes = np.where(best_pred == cluster_id)[0]
        predicted_values = data.iloc[cluster_genes, :].mean(axis=0).values

        plt.figure(figsize=(6,4))
        plt.plot(timestamps, true_values, 'o-', color='blue', label='Actual')
        plt.plot(timestamps, predicted_values, 'o--', color='red', label='Predicted')
        plt.title(f'Gene_{gene_idx}: Actual vs Predicted')
        plt.xlabel('Time point')
        plt.ylabel('Expression')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'gene_{gene_idx}_plot.png'))
        plt.close()