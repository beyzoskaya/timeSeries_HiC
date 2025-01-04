import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def temporal_pattern_loss(output, target, input_sequence):

    mse_loss = F.mse_loss(output, target)
    
    input_trend = input_sequence[:, :, 1:, :] - input_sequence[:, :, :-1, :]
    last_trend = input_sequence[:, :, -1, :] - input_sequence[:, :, -2, :]
    
    pred_trend = output - input_sequence[:, :, -1:, :]
    
    trend_loss = F.mse_loss(pred_trend, last_trend)
    
    return mse_loss + 0.1 * trend_loss

def change_magnitude_loss(output, target, input_sequence, alpha=1.0, beta=0.5):

    pred_loss = F.mse_loss(output, target)
    
    actual_change = target - input_sequence[:, :, -1:, :]
    pred_change = output - input_sequence[:, :, -1:, :]
    
    magnitude_loss = F.mse_loss(torch.abs(pred_change), torch.abs(actual_change))
    
    underpredict_penalty = torch.mean(
        torch.relu(torch.abs(actual_change) - torch.abs(pred_change))
    )
    
    return pred_loss + alpha * magnitude_loss + beta * underpredict_penalty