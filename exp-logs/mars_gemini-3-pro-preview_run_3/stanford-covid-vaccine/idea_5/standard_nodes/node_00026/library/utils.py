import os
import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error Loss.
    Calculates the MCRMSE over all target columns provided in the input.
    Used as the training objective to learn shared representations across all conditions.
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predicted values (Batch, Seq_Len_Pred, Channels)
            targets: Ground truth values (Batch, Seq_Len_Scored, Channels)
        """
        # Slice inputs to match the scored sequence length of targets if necessary
        # Targets are typically length 68, while model outputs might be length 107
        if inputs.shape[1] > targets.shape[1]:
            inputs = inputs[:, : targets.shape[1], :]

        # Calculate Squared Error
        mse = (inputs - targets) ** 2

        # Average over batch and sequence length dimensions to get MSE per column
        # Flatten batch and sequence dims -> (Batch * Seq_Len, Channels)
        # Then mean over dim 0 -> (Channels,)
        mse_per_col = torch.mean(mse.view(-1, mse.shape[-1]), dim=0)

        # RMSE per column
        rmse_per_col = torch.sqrt(mse_per_col)

        # Mean of RMSEs across columns to get the final scalar loss
        mcrmse = torch.mean(rmse_per_col)

        return mcrmse


def calculate_metric(preds, targets):
    """
    Calculates the competition metric (MCRMSE) specifically on the 3 scored columns:
    reactivity, deg_Mg_pH10, and deg_Mg_50C.

    Args:
        preds: Predictions (Batch, Seq_Len, 5) - torch.Tensor or np.ndarray
        targets: Ground Truth (Batch, Seq_Len_Scored, 5) - torch.Tensor or np.ndarray

    Returns:
        float: The MCRMSE score on the scored columns.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Slice predictions to match target length (typically 68)
    if preds.shape[1] > targets.shape[1]:
        preds = preds[:, : targets.shape[1], :]

    # Filter for the specific scored columns defined in Config
    # Indices: 0 (reactivity), 1 (deg_Mg_pH10), 3 (deg_Mg_50C)
    scored_indices = Config.SCORED_COLS_INDICES

    preds_scored = preds[:, :, scored_indices]
    targets_scored = targets[:, :, scored_indices]

    # Calculate MSE per column
    # Flatten batch and sequence dimensions to aggregate all predictions for a column
    preds_flat = preds_scored.reshape(-1, preds_scored.shape[-1])
    targets_flat = targets_scored.reshape(-1, targets_scored.shape[-1])

    # Mean Squared Error per column
    mse_per_col = np.mean((preds_flat - targets_flat) ** 2, axis=0)

    # Root Mean Squared Error per column
    rmse_per_col = np.sqrt(mse_per_col)

    # Average the RMSEs to get MCRMSE
    metric = np.mean(rmse_per_col)

    return float(metric)


def create_submission_file(preds, test_ids):
    """
    Formats predictions into the required CSV submission format.

    Args:
        preds: Predictions array of shape (N_samples, 107, 5).
        test_ids: List of sample IDs corresponding to the predictions.
    """
    # Ensure preds is numpy
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()

    submission_data = []

    # The columns required in the submission file
    # id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    for i, sample_id in enumerate(test_ids):
        sample_preds = preds[i]  # Shape (107, 5)

        # Iterate over all 107 positions
        for seq_pos in range(sample_preds.shape[0]):
            # Construct the id_seqpos identifier
            row_id = f"{sample_id}_{seq_pos}"

            # Get predicted values for this position
            # The order in preds matches Config.TARGET_COLS, which matches submission requirement
            row_values = sample_preds[seq_pos].tolist()

            submission_data.append([row_id] + row_values)

    # Define header
    columns = ["id_seqpos"] + Config.TARGET_COLS

    # Create DataFrame and save
    submission_df = pd.DataFrame(submission_data, columns=columns)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
