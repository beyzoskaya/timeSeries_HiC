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

def temporal_loss_for_projected_model(output, target,input_sequence, alpha=0.3, beta=0.2, gamma=0.2):
    """
    output: predicted expression values -->[1, 1, 1, 52] (52 unique genes!)
    target: compare the real labels which are actal expression values with outputs --> [1, 1, 1, 52]
    input sequence: input seqeunce to prediction --> [1, 32, 3, 52]
    """

    print(f"Output shape: {output.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Input sequence shape: {input_sequence.shape}")
    
    mse_loss = F.mse_loss(output, target)
    last_input = input_sequence[:, -1:, -1, :]

    true_change = target - last_input
    pred_change = output - last_input

    direction_match = torch.sign(true_change) * torch.sign(pred_change) # control the direction as a negative of positive
    direction_loss = -torch.mean(direction_match)

    magnitude_loss = F.mse_loss(F.mse_loss(torch.abs(true_change), torch.abs(pred_change)))

    def trend_correlation_loss(pred, target, sequence):
        seq_expressions = sequence[:, -1, :, :] # [1, 3, 52]

        pred_trend = torch.cat([seq_expressions, pred], dim=1)
        target_trend = torch.cat([seq_expressions, target], dim=1)

        pred_norm = (pred_trend - pred_trend.mean()) / (pred_trend.std() + 1e-8)
        target_norm = (target_trend - target_trend.mean()) / (target_trend.std() + 1e-8)
        
        return -torch.mean(pred_norm * target_norm)
    
    temp_loss = trend_correlation_loss(output, target, input_sequence[:, -1:, :, :])

    print(f"\nLoss Components:")
    print(f"MSE Loss: {mse_loss.item():.4f}")
    print(f"Direction Loss: {direction_loss.item():.4f}")
    print(f"Magnitude Loss: {magnitude_loss.item():.4f}")
    print(f"Temporal Loss: {temp_loss.item():.4f}")

    total_loss = mse_loss + alpha * direction_loss + beta * magnitude_loss + gamma * temp_loss

    return total_loss

