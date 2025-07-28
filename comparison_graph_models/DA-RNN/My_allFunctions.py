import torch
import numpy as np
from matplotlib import pyplot as plt
import collections
import typing
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as tf

import logging
import os

import typing
from typing import Tuple
import json

from torch import optim
from sklearn.preprocessing import StandardScaler
import joblib

import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TrainConfig(typing.NamedTuple):
    T: int
    train_size: int
    batch_size: int
    loss_func: typing.Callable

class TrainData(typing.NamedTuple):
    feats: np.ndarray
    targs: np.ndarray

DaRnnNet = collections.namedtuple("DaRnnNet", ["encoder", "decoder", "enc_opt", "dec_opt"])

def setup_log(tag='VOC_TOPICS'):
    logger = logging.getLogger(tag)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def save_or_show_plot(file_nm: str, save: bool):
    if save:
        plt.savefig(os.path.join(os.path.dirname(__file__), "plots", file_nm))
    else:
        plt.show()

def numpy_to_tvar(x):
    return Variable(torch.from_numpy(x).type(torch.FloatTensor).to(device))

def init_hidden(x, hidden_size: int):
    return Variable(torch.zeros(1, x.size(0), hidden_size))

def create_multi_gene_dataset(data, target_cols, seq_len, n_features=128):
    gene_names = target_cols
    n_genes = len(gene_names)
    
    all_samples = []
    all_targets = []
    all_gene_ids = []
    
    # Calculate correlations between all genes
    corr_matrix = data.corr()
    
    for gene_idx, gene_name in enumerate(gene_names):
        # Get top-n most correlated genes (excluding self)
        gene_corrs = corr_matrix[gene_name].abs().sort_values(ascending=False)
        top_genes = gene_corrs.head(n_features + 1).index.tolist()
        if gene_name in top_genes:
            top_genes.remove(gene_name)
        top_genes = top_genes[:n_features]
        
        # Create sequences using these features
        for t in range(seq_len, len(data)):
            # Features: history of top correlated genes
            sequence = data[top_genes].iloc[t-seq_len:t].values  # Shape: (seq_len, n_features)
            target = data[gene_name].iloc[t]
            
            all_samples.append(sequence)
            all_targets.append(target)
            all_gene_ids.append(gene_idx)
    
    return np.array(all_samples), np.array(all_targets), np.array(all_gene_ids)

# MODIFIED: Enhanced Encoder with gene embeddings
class Enhanced_Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, T: int, n_genes: int, gene_embed_dim: int = 16):
        super(Enhanced_Encoder, self).__init__()
        self.input_size = input_size  # This is 1
        self.hidden_size = hidden_size
        self.T = T
        self.gene_embed_dim = gene_embed_dim
        
        # Gene embedding layer
        self.gene_embedding = nn.Embedding(n_genes, gene_embed_dim)
        
        # Use input_size (1) for LSTM, not total_input_size
        self.lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        self.attn_linear = nn.Linear(in_features=2 * hidden_size + T - 1, out_features=1)

    def forward(self, input_data, gene_ids):
        batch_size = input_data.size(0)
        
        # Get gene embeddings and add them to input (instead of concatenate)
        gene_embeds = self.gene_embedding(gene_ids)  # (batch_size, 16)
        gene_embeds_expanded = gene_embeds.unsqueeze(1).repeat(1, self.T - 1, 1)  # (batch_size, T-1, 16)
        
        # Project gene embeddings to match input size and add
        gene_effect = gene_embeds_expanded[:, :, 0:1]  # Take first dimension only -> (batch_size, T-1, 1)
        enhanced_input = input_data + gene_effect  # Both are (batch_size, T-1, 1)
        
        # Initialize outputs with correct sizes
        input_weighted = Variable(torch.zeros(batch_size, self.T - 1, self.input_size))
        input_encoded = Variable(torch.zeros(batch_size, self.T - 1, self.hidden_size))
        
        hidden = init_hidden(enhanced_input, self.hidden_size)
        cell = init_hidden(enhanced_input, self.hidden_size)

        for t in range(self.T - 1):
            # Use self.input_size (1) for attention, not total_input_size
            x = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           enhanced_input.permute(0, 2, 1)), dim=2)
            
            x = self.attn_linear(x.view(-1, self.hidden_size * 2 + self.T - 1))
            attn_weights = tf.softmax(x.view(-1, self.input_size), dim=1)
            
            weighted_input = torch.mul(attn_weights, enhanced_input[:, t, :])
            
            self.lstm_layer.flatten_parameters()
            _, lstm_states = self.lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))
            hidden = lstm_states[0]
            cell = lstm_states[1]
            
            input_weighted[:, t, :] = weighted_input
            input_encoded[:, t, :] = hidden

        return attn_weights, input_weighted, input_encoded

class Enhanced_Decoder(nn.Module):
    def __init__(self, encoder_hidden_size: int, decoder_hidden_size: int, T: int, out_feats=1):
        super(Enhanced_Decoder, self).__init__()

        self.T = T
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.attn_layer = nn.Sequential(nn.Linear(2 * decoder_hidden_size + encoder_hidden_size,
                                                  encoder_hidden_size),
                                        nn.Tanh(),
                                        nn.Linear(encoder_hidden_size, 1))
        self.lstm_layer = nn.LSTM(input_size=out_feats, hidden_size=decoder_hidden_size)
        self.fc = nn.Linear(encoder_hidden_size + out_feats, out_feats)
        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, out_feats)

        self.fc.weight.data.normal_()

    def forward(self, input_encoded, y_history):
        hidden = init_hidden(input_encoded, self.decoder_hidden_size)
        cell = init_hidden(input_encoded, self.decoder_hidden_size)
        context = Variable(torch.zeros(input_encoded.size(0), self.encoder_hidden_size))

        for t in range(self.T - 1):
            x = torch.cat((hidden.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           input_encoded), dim=2)
            x = tf.softmax(
                    self.attn_layer(
                        x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size)
                    ).view(-1, self.T - 1),
                    dim=1)

            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :]
            y_tilde = self.fc(torch.cat((context, y_history[:, t]), dim=1))
            self.lstm_layer.flatten_parameters()
            _, lstm_output = self.lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))
            hidden = lstm_output[0]
            cell = lstm_output[1]

        return self.fc_final(torch.cat((hidden[0], context), dim=1))

logger = setup_log()
logger.info(f"Using computation device: {device}")

def preprocess_data(dat, col_names, scaler=None) -> Tuple[TrainData, StandardScaler]:
    if scaler is None:
        scaler = StandardScaler().fit(dat)
    proc_dat = scaler.transform(dat)

    mask = np.ones(proc_dat.shape[1], dtype=bool)
    dat_cols = list(dat.columns)
    for col_name in col_names:
        mask[dat_cols.index(col_name)] = False

    feats = proc_dat[:, mask]
    targs = proc_dat[:, ~mask]

    return TrainData(feats, targs), scaler

# NEW: DA-RNN with gene embeddings
def da_rnn_with_gene_embedding(train_data: TrainData, n_genes: int, n_targs: int, 
                               encoder_hidden_size=64, decoder_hidden_size=64,
                               T=10, learning_rate=0.01, batch_size=4):

    #train_cfg = TrainConfig(T, int(train_data.feats.shape[0] * 0.7), batch_size, nn.MSELoss())
    train_cfg = TrainConfig(T, len(train_data.feats) - T, batch_size, nn.MSELoss())
    logger.info(f"Training size: {train_cfg.train_size:d}.")

    enc_kwargs = {"input_size": train_data.feats.shape[2], "hidden_size": encoder_hidden_size, 
              "T": T, "n_genes": n_genes}
    encoder = Enhanced_Encoder(**enc_kwargs).to(device)
    
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "enc_kwargs.json"), "w") as fi:
        json.dump(enc_kwargs, fi, indent=4)

    dec_kwargs = {"encoder_hidden_size": encoder_hidden_size,
                  "decoder_hidden_size": decoder_hidden_size, "T": T, "out_feats": n_targs}
    decoder = Enhanced_Decoder(**dec_kwargs).to(device)
    with open(os.path.join("data", "dec_kwargs.json"), "w") as fi:
        json.dump(dec_kwargs, fi, indent=4)

    encoder_optimizer = optim.Adam(
        params=[p for p in encoder.parameters() if p.requires_grad],
        lr=learning_rate)
    decoder_optimizer = optim.Adam(
        params=[p for p in decoder.parameters() if p.requires_grad],
        lr=learning_rate)
    da_rnn_net = DaRnnNet(encoder, decoder, encoder_optimizer, decoder_optimizer)

    return train_cfg, da_rnn_net

def da_rnn(train_data: TrainData, n_targs: int, encoder_hidden_size=64, decoder_hidden_size=64,
           T=10, learning_rate=0.01, batch_size=4):

    train_cfg = TrainConfig(T, int(train_data.feats.shape[0] * 0.7), batch_size, nn.MSELoss())
    logger.info(f"Training size: {train_cfg.train_size:d}.")

    enc_kwargs = {"input_size": train_data.feats.shape[1], "hidden_size": encoder_hidden_size, "T": T}
    encoder = Encoder(**enc_kwargs).to(device)
    
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "enc_kwargs.json"), "w") as fi:
        json.dump(enc_kwargs, fi, indent=4)

    dec_kwargs = {"encoder_hidden_size": encoder_hidden_size,
                  "decoder_hidden_size": decoder_hidden_size, "T": T, "out_feats": n_targs}
    decoder = Decoder(**dec_kwargs).to(device)
    with open(os.path.join("data", "dec_kwargs.json"), "w") as fi:
        json.dump(dec_kwargs, fi, indent=4)

    encoder_optimizer = optim.Adam(
        params=[p for p in encoder.parameters() if p.requires_grad],
        lr=learning_rate)
    decoder_optimizer = optim.Adam(
        params=[p for p in decoder.parameters() if p.requires_grad],
        lr=learning_rate)
    da_rnn_net = DaRnnNet(encoder, decoder, encoder_optimizer, decoder_optimizer)

    return train_cfg, da_rnn_net

class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, T: int):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T

        self.lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        self.attn_linear = nn.Linear(in_features=2 * hidden_size + T - 1, out_features=1)

    def forward(self, input_data):
        input_weighted = Variable(torch.zeros(input_data.size(0), self.T - 1, self.input_size))
        input_encoded = Variable(torch.zeros(input_data.size(0), self.T - 1, self.hidden_size))
        hidden = init_hidden(input_data, self.hidden_size)
        cell = init_hidden(input_data, self.hidden_size)

        for t in range(self.T - 1):
            x = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           input_data.permute(0, 2, 1)), dim=2)
            x = self.attn_linear(x.view(-1, self.hidden_size * 2 + self.T - 1))
            attn_weights = tf.softmax(x.view(-1, self.input_size), dim=1)
            weighted_input = torch.mul(attn_weights, input_data[:, t, :])
            self.lstm_layer.flatten_parameters()
            _, lstm_states = self.lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))
            hidden = lstm_states[0]
            cell = lstm_states[1]
            input_weighted[:, t, :] = weighted_input
            input_encoded[:, t, :] = hidden

        return attn_weights, input_weighted, input_encoded

class Decoder(nn.Module):
    def __init__(self, encoder_hidden_size: int, decoder_hidden_size: int, T: int, out_feats=1):
        super(Decoder, self).__init__()

        self.T = T
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.attn_layer = nn.Sequential(nn.Linear(2 * decoder_hidden_size + encoder_hidden_size,
                                                  encoder_hidden_size),
                                        nn.Tanh(),
                                        nn.Linear(encoder_hidden_size, 1))
        self.lstm_layer = nn.LSTM(input_size=out_feats, hidden_size=decoder_hidden_size)
        self.fc = nn.Linear(encoder_hidden_size + out_feats, out_feats)
        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, out_feats)

        self.fc.weight.data.normal_()

    def forward(self, input_encoded, y_history):
        hidden = init_hidden(input_encoded, self.decoder_hidden_size)
        cell = init_hidden(input_encoded, self.decoder_hidden_size)
        context = Variable(torch.zeros(input_encoded.size(0), self.encoder_hidden_size))

        for t in range(self.T - 1):
            x = torch.cat((hidden.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           input_encoded), dim=2)
            x = tf.softmax(
                    self.attn_layer(
                        x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size)
                    ).view(-1, self.T - 1),
                    dim=1)

            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :]
            y_tilde = self.fc(torch.cat((context, y_history[:, t]), dim=1))
            self.lstm_layer.flatten_parameters()
            _, lstm_output = self.lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))
            hidden = lstm_output[0]
            cell = lstm_output[1]

        return self.fc_final(torch.cat((hidden[0], context), dim=1))

# NEW: Training function with gene IDs
def train_with_gene_ids(net: DaRnnNet, train_data: TrainData, gene_ids: np.ndarray, 
                       t_cfg: TrainConfig, n_epochs=10, save_plots=False):
    iter_per_epoch = int(np.ceil(t_cfg.train_size * 1. / t_cfg.batch_size))
    iter_losses = []
    iter_attn = []
    epoch_losses = []
    logger.info(f"Iterations per epoch: {t_cfg.train_size * 1. / t_cfg.batch_size:3.3f} ~ {iter_per_epoch:d}.")

    n_iter = 0

    for e_i in range(n_epochs):
        #perm_idx = np.random.permutation(t_cfg.train_size - t_cfg.T)
        max_idx = len(train_data.feats) - t_cfg.T - 1  # Leave room for T-length sequences
        perm_idx = np.random.permutation(max_idx)

        for t_i in range(0, t_cfg.train_size, t_cfg.batch_size):
            batch_idx = perm_idx[t_i:(t_i + t_cfg.batch_size)]
            feats, y_history, y_target, batch_gene_ids = prep_train_data_with_gene_ids(
                batch_idx, t_cfg, train_data, gene_ids)

            loss, attn = train_iteration_with_gene_ids(net, t_cfg.loss_func, feats, y_history, y_target, batch_gene_ids)
            iter_losses.append(loss)
            iter_attn.append(attn)
            n_iter += 1

            adjust_learning_rate(net, n_iter)
        epoch_losses.append(loss)

    return iter_losses, epoch_losses, iter_attn

def prep_train_data_with_gene_ids(batch_idx: np.ndarray, t_cfg: TrainConfig, train_data: TrainData, gene_ids: np.ndarray):
    n_features = train_data.feats.shape[2]  # 20
    
    feats = np.zeros((len(batch_idx), t_cfg.T - 1, n_features))
    y_history = np.zeros((len(batch_idx), t_cfg.T - 1, 1))
    y_target = train_data.targs[batch_idx + t_cfg.T]
    batch_gene_ids = gene_ids[batch_idx + t_cfg.T]

    for b_i, b_idx in enumerate(batch_idx):
        # Extract the right time window
        feats[b_i] = train_data.feats[b_idx, :t_cfg.T-1, :]  # Use specific time indices
        y_history[b_i, :, 0] = train_data.targs[b_idx:b_idx + t_cfg.T - 1, 0]

    return feats, y_history, y_target, batch_gene_ids

# NEW: Training iteration with gene IDs
def train_iteration_with_gene_ids(t_net: DaRnnNet, loss_func: typing.Callable, X, y_history, y_target, gene_ids):
    t_net.enc_opt.zero_grad()
    t_net.dec_opt.zero_grad()

    gene_ids_tensor = torch.from_numpy(gene_ids).long().to(device)
    attn_w, input_weighted, input_encoded = t_net.encoder(numpy_to_tvar(X), gene_ids_tensor)
    y_pred = t_net.decoder(input_encoded, numpy_to_tvar(y_history))

    y_true = numpy_to_tvar(y_target)
    loss = loss_func(y_pred, y_true)
    loss.backward()

    t_net.enc_opt.step()
    t_net.dec_opt.step()

    return loss.item(), attn_w

# NEW: Prediction with gene IDs
def predict_with_gene_ids(model: DaRnnNet, test_X, test_gene_ids, target_scaler):
    model.encoder.eval()
    model.decoder.eval()
    
    predictions = []
    
    with torch.no_grad():
        batch_size = 32  # Process in batches
        n_samples = len(test_X)
        
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_X = test_X[i:batch_end]
            batch_gene_ids = test_gene_ids[i:batch_end]
            
            # Create dummy y_history (zeros)
            batch_y_history = np.zeros((len(batch_X), batch_X.shape[1], 1))
            
            gene_ids_tensor = torch.from_numpy(batch_gene_ids).long().to(device)
            attn_w, input_weighted, input_encoded = model.encoder(numpy_to_tvar(batch_X), gene_ids_tensor)
            batch_pred = model.decoder(input_encoded, numpy_to_tvar(batch_y_history))
            
            predictions.extend(batch_pred.cpu().data.numpy())
    
    model.encoder.train()
    model.decoder.train()
    
    # Convert back to original scale
    predictions = np.array(predictions).reshape(-1, 1)
    predictions_original = target_scaler.inverse_transform(predictions).flatten()
    
    return predictions_original

def train(net: DaRnnNet, train_data: TrainData, t_cfg: TrainConfig, n_epochs=10, save_plots=False, check_train_epoch=False):
    iter_per_epoch = int(np.ceil(t_cfg.train_size * 1. / t_cfg.batch_size))
    iter_losses = []
    attn_losses = []
    iter_attn = []
    epoch_losses = []
    logger.info(f"Iterations per epoch: {t_cfg.train_size * 1. / t_cfg.batch_size:3.3f} ~ {iter_per_epoch:d}.")

    n_iter = 0

    for e_i in range(n_epochs):
        perm_idx = np.random.permutation(t_cfg.train_size - t_cfg.T)

        for t_i in range(0, t_cfg.train_size, t_cfg.batch_size):
            batch_idx = perm_idx[t_i:(t_i + t_cfg.batch_size)]
            feats, y_history, y_target = prep_train_data(batch_idx, t_cfg, train_data)

            loss = train_iteration(net, t_cfg.loss_func, feats, y_history, y_target)[0]
            attn = train_iteration(net, t_cfg.loss_func, feats, y_history, y_target)[1]
            iter_losses.append(loss)
            iter_attn.append(attn)
            n_iter += 1

            adjust_learning_rate(net, n_iter)
        epoch_losses.append(loss)

    return iter_losses, epoch_losses, iter_attn

def prep_train_data(batch_idx: np.ndarray, t_cfg: TrainConfig, train_data: TrainData):
    feats = np.zeros((len(batch_idx), t_cfg.T - 1, train_data.feats.shape[1]))
    y_history = np.zeros((len(batch_idx), t_cfg.T - 1, train_data.targs.shape[1]))
    y_target = train_data.targs[batch_idx + t_cfg.T]

    for b_i, b_idx in enumerate(batch_idx):
        b_slc = slice(b_idx, b_idx + t_cfg.T - 1)
        feats[b_i, :, :] = train_data.feats[b_slc, :]
        y_history[b_i, :] = train_data.targs[b_slc]

    return feats, y_history, y_target

def adjust_learning_rate(net: DaRnnNet, n_iter: int):
    if n_iter % 10000 == 0 and n_iter > 0:
        for enc_params, dec_params in zip(net.enc_opt.param_groups, net.dec_opt.param_groups):
            enc_params['lr'] = enc_params['lr'] * 0.9
            dec_params['lr'] = dec_params['lr'] * 0.9

def train_iteration(t_net: DaRnnNet, loss_func: typing.Callable, X, y_history, y_target):
    t_net.enc_opt.zero_grad()
    t_net.dec_opt.zero_grad()

    attn_w, input_weighted, input_encoded = t_net.encoder(numpy_to_tvar(X))
    y_pred = t_net.decoder(input_encoded, numpy_to_tvar(y_history))

    y_true = numpy_to_tvar(y_target)
    loss = loss_func(y_pred, y_true)
    loss.backward()

    t_net.enc_opt.step()
    t_net.dec_opt.step()

    return loss.item(), attn_w

def my_predict(model: DaRnnNet, prep: TrainData, T: int, TimeFuture: int):
    model.encoder.eval()
    model.decoder.eval()
    
    batch_size = 1
    out_size = prep.targs.shape[1]
    
    max_predictions = len(prep.feats) - T + 1
    actual_predictions = min(TimeFuture, max_predictions)
    
    if actual_predictions <= 0:
        print(f"Warning: Not enough data to make predictions. Need at least {T} time points.")
        return np.array([])
    
    predictions = []
    
    with torch.no_grad():
        for i in range(actual_predictions):
            feat_window = prep.feats[i:i+T-1].reshape(1, T-1, -1)
            targ_window = prep.targs[i:i+T-1].reshape(1, T-1, -1)
            
            feat_tensor = numpy_to_tvar(feat_window)
            targ_tensor = numpy_to_tvar(targ_window)
            
            attn_w, input_weighted, input_encoded = model.encoder(feat_tensor)
            pred = model.decoder(input_encoded, targ_tensor)
            
            predictions.append(pred.cpu().data.numpy()[0])
    
    model.encoder.train()
    model.decoder.train()
    
    return np.array(predictions)