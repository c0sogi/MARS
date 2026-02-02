import torch
import numpy as np
import os
import random
from library.config import setup_reproducibility


def seed_everything(seed: int):
    """
    Sets random seeds for reproducibility across Python, NumPy, and PyTorch.
    Delegates to the library configuration function.

    Args:
        seed (int): The seed value to set.
    """
    setup_reproducibility(seed)


def calculate_mcrmse(preds, targets, scored_indices=None):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    This function computes the RMSE for each target column independently and then
    returns the average RMSE across columns. It ensures that the metric is calculated
    globally over the entire dataset (or provided batch) rather than averaging
    sample-wise or batch-wise errors, which can introduce bias.

    Args:
        preds (np.ndarray or torch.Tensor): Predicted values.
            Expected shape: (N_samples, Seq_Len, N_channels) or (N_samples, N_channels).
        targets (np.ndarray or torch.Tensor): Ground truth values.
            Expected shape: Same as preds.
        scored_indices (list of int, optional): A list of column indices to include
            in the metric calculation. If None, all columns are used.
            Example: [0, 1, 3] for reactivity, deg_Mg_pH10, deg_Mg_50C.

    Returns:
        float: The calculated MCRMSE score.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    preds = np.asarray(preds)
    targets = np.asarray(targets)

    # Validate shapes
    if preds.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch in metric calculation: preds {preds.shape} vs targets {targets.shape}"
        )

    # Determine number of channels (last dimension)
    num_channels = preds.shape[-1]

    # Flatten all dimensions except the channel dimension
    # This aggregates all (sample, position) pairs for each target type
    preds_flat = preds.reshape(-1, num_channels)
    targets_flat = targets.reshape(-1, num_channels)

    # Filter for specific columns if indices are provided
    if scored_indices is not None:
        preds_flat = preds_flat[:, scored_indices]
        targets_flat = targets_flat[:, scored_indices]

    # Compute Mean Squared Error (MSE) for each column
    # axis=0 aggregates over all flattened samples/positions
    mse_per_column = np.mean((targets_flat - preds_flat) ** 2, axis=0)

    # Compute Root Mean Squared Error (RMSE) for each column
    rmse_per_column = np.sqrt(mse_per_column)

    # Compute the Mean of the column-wise RMSEs
    mcrmse = np.mean(rmse_per_column)

    return float(mcrmse)
