import os
import random
import numpy as np
import torch
from library import config


def set_seed(seed: int = config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Returns the appropriate PyTorch device (CUDA if available, else CPU).

    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_mcrmse(
    preds,
    targets,
    scored_indices=config.SCORED_COLS_INDICES,
    seq_scored=config.SEQ_SCORED,
):
    """
    Computes the Mean Columnwise Root Mean Squared Error (MCRMSE) for the validation set.

    This function calculates the RMSE for each scored column globally across all samples
    and sequence positions, then averages these RMSE values. This avoids the bias
    introduced by averaging RMSEs calculated on smaller batches.

    Args:
        preds (np.ndarray or torch.Tensor): Predictions with shape (N, L, 5).
                                            Can be longer than seq_scored.
        targets (np.ndarray or torch.Tensor): Ground truth with shape (N, L_target, 5).
                                              L_target must be at least seq_scored.
        scored_indices (list): Indices of the columns to score. Defaults to config.SCORED_COLS_INDICES.
        seq_scored (int): The number of sequence positions to score (starting from 0).
                          Defaults to config.SEQ_SCORED.

    Returns:
        float: The computed MCRMSE value.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Ensure inputs are at least as long as the scored sequence length
    # Slicing: Select all samples, first 'seq_scored' positions, and all columns initially
    preds_sliced = preds[:, :seq_scored, :]
    targets_sliced = targets[:, :seq_scored, :]

    # Filter for the specific columns used in the competition metric
    # Typically: reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (3)
    preds_selected = preds_sliced[:, :, scored_indices]
    targets_selected = targets_sliced[:, :, scored_indices]

    # Compute squared differences
    squared_diff = (preds_selected - targets_selected) ** 2

    # Compute Mean Squared Error (MSE) per column
    # We average over samples (axis 0) and sequence positions (axis 1)
    # Result shape: (num_scored_cols,)
    mse_per_col = np.mean(squared_diff, axis=(0, 1))

    # Compute RMSE per column
    rmse_per_col = np.sqrt(mse_per_col)

    # Compute MCRMSE: Average of the column-wise RMSEs
    mcrmse = np.mean(rmse_per_col)

    return float(mcrmse)
