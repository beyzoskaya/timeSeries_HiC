#!/usr/bin/env python
# coding: utf-8

# # Single DA-RNN model for predicting all genes
# Updated: 07/2025

import os
import re
import time
import json
import pickle
import joblib
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as tf
import torch.multiprocessing as mp
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr

from My_allFunctions import *

# ---------------------------
# Utility functions (keep original ones)
# ---------------------------
def txt_to_csv(txtFile, csvFile, ncols=None):
    """Convert txt file with tab separator to csv and return DataFrame + cols"""
    df = pd.read_csv(txtFile, sep='\t')
    df.to_csv(csvFile, index=False)
    print(f"Converted {txtFile} -> {csvFile}")
    if ncols:
        cols = df.columns[ncols[0]:].tolist()
    else:
        cols = df.columns.tolist()
    return df, cols

def findInteraction(attn, n_epochs, plotAll=False):
    """Aggregate attention vectors over last epoch"""
    att = [np.array(i.detach().numpy()).sum(axis=0) for i in attn]
    totAtt = np.zeros(np.shape(att[0]))
    batch_x_epoch = int(len(attn) / n_epochs)
    for i in att[-batch_x_epoch:]:
        totAtt += i[0]
    if plotAll:
        for i in att:
            plt.plot(i[0])
        plt.figure(2)
        plt.plot(totAtt[0] / batch_x_epoch)
    return totAtt / batch_x_epoch

# ---------------------------
# NEW: Single model training function
# ---------------------------
def train_all_genes_single_model(csvFile, cols, da_rnn_kwargs, sub, n_epochs):
    """Train single DA-RNN model for all genes with multivariate input"""
    
    raw_data = pd.read_csv(csvFile, usecols=cols)
    print(f"Loaded data shape: {raw_data.shape}")
    
    # Remove time column if exists
    if 'time' in raw_data.columns:
        raw_data = raw_data.drop('time', axis=1)
        print("Removed time column")
    
    target_cols = [col for col in raw_data.columns]
    print(f"Training single model for {len(target_cols)} genes")
    
    # Split train/test
    train_size = int(len(raw_data) * 0.8)
    train_data = raw_data.iloc[:train_size]
    test_data = raw_data.iloc[train_size:]
    
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    
    T = da_rnn_kwargs["T"]
    if len(train_data) < T + 5:
        print(f"Error: Not enough training data. Need at least {T + 5} samples, got {len(train_data)}")
        return 0.0, 0.0, {}
    
    # Create datasets with multivariate features (20 most correlated genes per target)
    n_features = 128  # Number of correlated genes to use as features
    train_X, train_y, train_gene_ids = create_multi_gene_dataset(train_data, target_cols, T, n_features)
    test_X, test_y, test_gene_ids = create_multi_gene_dataset(test_data, target_cols, T, n_features)
    
    print(f"Training samples: {len(train_X)}, Test samples: {len(test_X)}")
    print(f"Sample input shape: {train_X[0].shape}")  # Should be (T, n_features) = (5, 20)
    print(f"Number of features per sample: {n_features}")
    
    # Scale data - need to handle multivariate features properly
    print("Scaling data...")
    
    # Reshape for scaling: (n_samples * T, n_features)
    train_X_reshaped = train_X.reshape(-1, n_features)
    test_X_reshaped = test_X.reshape(-1, n_features)
    
    # Fit scaler on training data
    scaler = StandardScaler()
    train_X_scaled_flat = scaler.fit_transform(train_X_reshaped)
    test_X_scaled_flat = scaler.transform(test_X_reshaped)
    
    # Reshape back to original format: (n_samples, T, n_features)
    train_X_scaled = train_X_scaled_flat.reshape(-1, T, n_features)
    test_X_scaled = test_X_scaled_flat.reshape(-1, T, n_features)
    
    # Scale targets
    target_scaler = StandardScaler()
    train_y_scaled = target_scaler.fit_transform(train_y.reshape(-1, 1)).flatten()
    test_y_scaled = target_scaler.transform(test_y.reshape(-1, 1)).flatten()
    
    print(f"Scaled training data shape: {train_X_scaled.shape}")  # Should be (19764, 5, 20)
    
    # Create train data structure - using first T-1 time points as input
    prep_train = TrainData(train_X_scaled, train_y_scaled.reshape(-1, 1))
    
    print(f"TrainData features shape: {prep_train.feats.shape}")  # Should be (19764, 4, 20)
    print(f"TrainData targets shape: {prep_train.targs.shape}")   # Should be (19764, 1)
    
    # Build model with gene embeddings
    n_genes = len(target_cols)
    print(f"Building model with {n_genes} gene embeddings and {n_features} input features...")
    
    try:
        config, model = da_rnn_with_gene_embedding(prep_train, n_genes=n_genes, 
                                                  n_targs=1, learning_rate=0.001, **da_rnn_kwargs)
        print("Model created successfully")
        print(f"Model input size: {model.encoder.input_size}")  
    except Exception as e:
        print(f"Error creating model: {e}")
        return 0.0, 0.0, {}
    
    # Train model
    print("Starting training...")
    try:
        iter_loss, epoch_loss, attn = train_with_gene_ids(model, prep_train, train_gene_ids, 
                                                         config, n_epochs=n_epochs)
        print(f"Training completed. Final loss: {epoch_loss[-1]:.6f}")
    except Exception as e:
        print(f"Error during training: {e}")
        return 0.0, 0.0, {}
    
    # Predict on test set
    print("Making predictions on test set...")
    try:
        test_input = test_X_scaled[:, :-1, :]  # Use first T-1 time points
        test_predictions = predict_with_gene_ids(model, test_input, test_gene_ids, target_scaler)
        print(f"Generated {len(test_predictions)} predictions")
    except Exception as e:
        print(f"Error during prediction: {e}")
        return 0.0, 0.0, {}
    
    # Calculate correlations per gene
    print("Calculating correlations...")
    gene_correlations = {}
    all_spearman = []
    all_pearson = []
    
    # Get original scale test targets
    test_y_original = target_scaler.inverse_transform(test_y_scaled.reshape(-1, 1)).flatten()
    
    for gene_idx, gene_name in enumerate(target_cols):
        # Get predictions and true values for this gene
        mask = test_gene_ids == gene_idx
        if np.sum(mask) < 3:  # Need at least 3 samples
            continue
            
        gene_true = test_y_original[mask]
        gene_pred = test_predictions[mask]
        
        # Remove any NaN or infinite values
        valid_mask = np.isfinite(gene_true) & np.isfinite(gene_pred)
        gene_true_clean = gene_true[valid_mask]
        gene_pred_clean = gene_pred[valid_mask]
        
        if len(gene_true_clean) < 3:
            continue
        
        # Check for constant arrays
        if np.std(gene_true_clean) < 1e-10 or np.std(gene_pred_clean) < 1e-10:
            continue
        
        # Calculate correlations
        try:
            s_corr, s_pval = spearmanr(gene_true_clean, gene_pred_clean)
            p_corr, p_pval = pearsonr(gene_true_clean, gene_pred_clean)
            
            if not np.isnan(s_corr) and not np.isnan(p_corr):
                gene_correlations[gene_name] = {
                    'spearman': s_corr, 
                    'pearson': p_corr,
                    'n_samples': len(gene_true_clean)
                }
                all_spearman.append(s_corr)
                all_pearson.append(p_corr)
                
        except Exception as e:
            print(f"Error calculating correlation for {gene_name}: {e}")
            continue
    
    # Calculate mean correlations
    mean_spearman = np.mean(all_spearman) if all_spearman else 0.0
    mean_pearson = np.mean(all_pearson) if all_pearson else 0.0
    
    print(f"\n=== Results ===")
    print(f"Valid genes with correlations: {len(all_spearman)}/{len(target_cols)}")
    print(f"Mean Spearman: {mean_spearman:.4f} (std: {np.std(all_spearman):.4f})")
    print(f"Mean Pearson: {mean_pearson:.4f} (std: {np.std(all_pearson):.4f})")
    
    if len(all_spearman) > 0:
        print(f"Spearman range: [{np.min(all_spearman):.4f}, {np.max(all_spearman):.4f}]")
        print(f"Pearson range: [{np.min(all_pearson):.4f}, {np.max(all_pearson):.4f}]")
    
    # Save results
    results_dir = os.path.join(sub, 'single_model_results_multivariate')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save summary
    summary_df = pd.DataFrame([{
        'mean_spearman': mean_spearman,
        'mean_pearson': mean_pearson,
        'std_spearman': np.std(all_spearman) if all_spearman else 0,
        'std_pearson': np.std(all_pearson) if all_pearson else 0,
        'valid_genes': len(all_spearman),
        'total_genes': len(target_cols),
        'n_features': n_features
    }])
    summary_df.to_csv(os.path.join(results_dir, 'summary.csv'), index=False)
    
    # Save per-gene results
    gene_results_df = pd.DataFrame([
        {'gene': gene, 'spearman': data['spearman'], 'pearson': data['pearson'], 'n_samples': data['n_samples']}
        for gene, data in gene_correlations.items()
    ])
    gene_results_df.to_csv(os.path.join(results_dir, 'per_gene_results.csv'), index=False)
    
    print(f"Results saved to {results_dir}")
    
    return mean_spearman, mean_pearson, gene_correlations

# ---------------------------
# Main script
# ---------------------------

if __name__ == "__main__":
    subfolders = ['./DATA/MyDataset']
    FileListtxt, FileListcsv, subList = [], [], []

    for sub in subfolders:
        print(f'----------- {sub} ------------')
        for file in os.listdir(sub):
            if 'Protein' in file and 'txt' in file:
                print(file)
                subList.append(sub)
                FileListtxt.append(os.path.join(sub, file))
                FileListcsv.append(os.path.join(sub, file.replace('txt', 'csv')))

    n_epochs = 50 
    da_rnn_kwargs = {"batch_size": 16} 
    
    print(f'Number of epochs: {n_epochs}')
    print(f'Number of GRNs: {len(FileListtxt)}')

    all_results = []

    for i, (txtFile, csvFile, sub) in enumerate(zip(FileListtxt, FileListcsv, subList)):
        print(f'\n=== Training GRN {i+1}/{len(FileListtxt)} with Single Model ===')
        start_time = time.time()

        try:
            timelist = np.loadtxt(txtFile, skiprows=1)[:, 0]
            n_timepoints = len(timelist)
            print(f'Time points: {n_timepoints}')
          
            max_T = min(5, n_timepoints//6)  
            T = max(3, max_T)  # At least 3
            print(f"Setting input window T = {T}")
            
            da_rnn_kwargs["T"] = T
            
        except Exception as e:
            print(f"Error loading time data: {e}")
            continue

        if not os.path.exists(csvFile):
            try:
                _, cols = txt_to_csv(txtFile, csvFile, ncols=[0])
            except Exception as e:
                print(f"Error converting file: {e}")
                continue
        else:
            try:
                cols = pd.read_csv(csvFile).columns.tolist()
            except Exception as e:
                print(f"Error reading CSV: {e}")
                continue

        print(f'Total columns: {len(cols)}')

        # Train single model for all genes
        try:
            mean_s, mean_p, gene_results = train_all_genes_single_model(
                csvFile, cols, da_rnn_kwargs, sub, n_epochs)
            
            all_results.append({
                'grn': i+1,
                'file': csvFile,
                'mean_spearman': mean_s,
                'mean_pearson': mean_p,
                'n_genes': len([col for col in cols if col.lower() != 'time']),
                'valid_predictions': len(gene_results)
            })
            
        except Exception as e:
            print(f"Error training GRN {i+1}: {e}")
            all_results.append({
                'grn': i+1,
                'file': csvFile,
                'mean_spearman': 0.0,
                'mean_pearson': 0.0,
                'n_genes': 0,
                'valid_predictions': 0
            })

        elapsed_time = time.time() - start_time
        print(f'Training time: {elapsed_time:.2f}s')

    print(f"\n=== FINAL SUMMARY ===")
    if all_results:
        valid_results = [r for r in all_results if r['mean_spearman'] > 0]
        
        if valid_results:
            overall_spearman = [r['mean_spearman'] for r in valid_results]
            overall_pearson = [r['mean_pearson'] for r in valid_results]
            
            print(f"Successfully processed: {len(valid_results)}/{len(all_results)} GRNs")
            print(f"Overall Mean Spearman: {np.mean(overall_spearman):.4f} ± {np.std(overall_spearman):.4f}")
            print(f"Overall Mean Pearson: {np.mean(overall_pearson):.4f} ± {np.std(overall_pearson):.4f}")
            
            results_df = pd.DataFrame(all_results)
            results_df.to_csv('./final_single_model_results.csv', index=False)
            print("Results saved to './final_single_model_results.csv'")
            
            your_spearman = 0.67
            your_pearson = 0.68
            baseline_spearman = np.mean(overall_spearman)
            baseline_pearson = np.mean(overall_pearson)
            
            print(f"\n=== COMPARISON ===")
            print(f"Your Model    - Spearman: {your_spearman:.3f}, Pearson: {your_pearson:.3f}")
            print(f"DA-RNN Baseline - Spearman: {baseline_spearman:.3f}, Pearson: {baseline_pearson:.3f}")
            print(f"Improvement   - Spearman: +{your_spearman - baseline_spearman:.3f}, Pearson: +{your_pearson - baseline_pearson:.3f}")
            
        else:
            print("No valid results obtained!")
    
    print("\n=== Training Complete ===")