import os
import random
import numpy as np
import torch
from library.config import Config, setup_reproducibility


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility.
    Wraps the setup_reproducibility function from library.config.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    setup_reproducibility(seed)


def get_device():
    """
    Returns the appropriate torch device (CUDA if available, else CPU).

    Returns:
        torch.device: The device to be used for computation.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mcrmse_loss(y_pred, y_true):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) loss.
    This function is differentiable and suitable for use as a training loss.

    Args:
        y_pred (torch.Tensor): Predicted values. Shape (Batch, Seq, Channels) or (Batch, Channels).
        y_true (torch.Tensor): Ground truth values. Shape matches y_pred.

    Returns:
        torch.Tensor: Scalar loss value representing the mean of RMSEs across columns.
    """
    # Ensure shapes match
    assert (
        y_pred.shape == y_true.shape
    ), f"Shape mismatch: {y_pred.shape} vs {y_true.shape}"

    # Flatten to (N, num_targets) to handle both (B, S, C) and (B, C) cases uniformly
    # This aggregates over batch and sequence dimensions while keeping channels separate
    num_targets = y_pred.shape[-1]
    y_pred_flat = y_pred.view(-1, num_targets)
    y_true_flat = y_true.view(-1, num_targets)

    # Compute MSE per column (dim=0 aggregates over all flattened samples)
    mse = torch.mean((y_pred_flat - y_true_flat) ** 2, dim=0)

    # Compute RMSE per column
    # Add a small epsilon to avoid NaN gradients if mse is exactly 0
    rmse = torch.sqrt(mse + 1e-8)

    # Average RMSE across columns to get MCRMSE
    loss = torch.mean(rmse)

    return loss


def compute_mcrmse_numpy(y_pred, y_true):
    """
    Computes MCRMSE using NumPy arrays.
    Useful for calculating the final validation metric on the CPU after
    accumulating predictions from the entire validation set.

    Args:
        y_pred (np.ndarray): Predictions.
        y_true (np.ndarray): Ground truth.

    Returns:
        float: MCRMSE value.
    """
    # Ensure shapes match
    assert (
        y_pred.shape == y_true.shape
    ), f"Shape mismatch: {y_pred.shape} vs {y_true.shape}"

    num_targets = y_pred.shape[-1]
    y_pred_flat = y_pred.reshape(-1, num_targets)
    y_true_flat = y_true.reshape(-1, num_targets)

    # MSE per column
    mse = np.mean((y_pred_flat - y_true_flat) ** 2, axis=0)

    # RMSE per column
    rmse = np.sqrt(mse)

    # Mean of RMSEs
    return np.mean(rmse)
