import os
import numpy as np
import torch
from library.config import set_seed


def seed_everything(seed=42):
    """
    Sets random seeds for reproducibility across random, numpy, and torch.
    Wraps the implementation from library.config to ensure consistency.

    Args:
        seed (int): The seed value to use.
    """
    set_seed(seed)


def calculate_global_mcrmse(preds, targets, seq_scored=68):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) on the global set.

    This function strictly adheres to the evaluation protocol:
    1. Slices predictions and targets to the first 'seq_scored' positions.
    2. Aggregates squared errors globally across all samples and positions.
    3. Computes RMSE for each column.
    4. Returns the mean of the column RMSEs.

    Args:
        preds (np.ndarray or torch.Tensor): Predicted values of shape (N, SeqLen, NumTargets).
        targets (np.ndarray or torch.Tensor): Ground truth values of shape (N, SeqLen, NumTargets).
        seq_scored (int): The number of sequence positions to include in the scoring (default: 68).

    Returns:
        float: The calculated MCRMSE score.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Verify shapes
    if preds.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: preds {preds.shape} vs targets {targets.shape}"
        )

    if preds.shape[1] < seq_scored:
        raise ValueError(
            f"Sequence length ({preds.shape[1]}) is smaller than seq_scored ({seq_scored})"
        )

    # Slice the data to the scored sequence length
    # Shape becomes: (N, seq_scored, NumTargets)
    preds_sliced = preds[:, :seq_scored, :]
    targets_sliced = targets[:, :seq_scored, :]

    # Calculate Squared Error
    # Shape: (N, seq_scored, NumTargets)
    squared_error = (targets_sliced - preds_sliced) ** 2

    # Calculate Mean Squared Error (MSE) per column
    # We average over axis 0 (samples) and axis 1 (sequence positions) to get global MSE per target
    # Shape: (NumTargets,)
    mse_per_column = np.mean(squared_error, axis=(0, 1))

    # Calculate Root Mean Squared Error (RMSE) per column
    rmse_per_column = np.sqrt(mse_per_column)

    # Calculate MCRMSE (Mean of RMSEs)
    mcrmse = np.mean(rmse_per_column)

    return float(mcrmse)
