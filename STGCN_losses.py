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

def temporal_loss_for_projected_model(output, target, input_sequence, alpha=0.3, beta=0.2, gamma=0.2):
   
    print(f"Output shape: {output.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Input sequence shape: {input_sequence.shape}")
    
    # Base MSE loss for prediction accuracy
    mse_loss = F.mse_loss(output, target)
    
    # Get expression values (last feature) from input sequence
    input_expressions = input_sequence[:, -1, :, :]  # [1, 3, 52]
    last_input = input_expressions[:, -1:, :]  # [1, 1, 52]
    
    # Reshape output and target to match
    output_reshaped = output.squeeze(1)  # [1, 1, 52]
    target_reshaped = target.squeeze(1)  # [1, 1, 52]
    
    # Calculate changes
    true_change = target_reshaped - last_input
    pred_change = output_reshaped - last_input
    
    # Direction loss
    direction_match = torch.sign(true_change) * torch.sign(pred_change)
    direction_loss = -torch.mean(direction_match)
    
    # Magnitude loss
    magnitude_loss = F.mse_loss(torch.abs(true_change), torch.abs(pred_change))
    
    # Temporal correlation using input sequence trend
    def trend_correlation_loss(pred, target, sequence_expr):
        # Sequence_expr is [1, 3, 52]
        # Add prediction and target [1, 1, 52] to make [1, 4, 52]
        pred_trend = torch.cat([sequence_expr, pred], dim=1)
        target_trend = torch.cat([sequence_expr, target], dim=1)
        
        # Normalize across time dimension (dim=1)
        pred_norm = (pred_trend - pred_trend.mean(dim=1, keepdim=True)) / (pred_trend.std(dim=1, keepdim=True) + 1e-8)
        target_norm = (target_trend - target_trend.mean(dim=1, keepdim=True)) / (target_trend.std(dim=1, keepdim=True) + 1e-8)
        
        return -torch.mean(torch.sum(pred_norm * target_norm, dim=1))
    
    temp_loss = trend_correlation_loss(output_reshaped, target_reshaped, input_expressions)
    
    print(f"\nLoss Components:")
    print(f"MSE Loss: {mse_loss.item():.4f}")
    print(f"Direction Loss: {direction_loss.item():.4f}")
    print(f"Magnitude Loss: {magnitude_loss.item():.4f}")
    print(f"Temporal Loss: {temp_loss.item():.4f}")
    
    total_loss = mse_loss + alpha * direction_loss + beta * magnitude_loss + gamma * temp_loss
    
    return total_loss
