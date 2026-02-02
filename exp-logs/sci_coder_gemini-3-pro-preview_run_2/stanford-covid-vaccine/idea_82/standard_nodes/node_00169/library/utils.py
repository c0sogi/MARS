import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_global_rmse(
    predictions, targets, scored_length=68, scored_cols_indices=None
):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) for the scored region.

    Args:
        predictions (torch.Tensor or np.ndarray): Predicted values of shape (N, L, 5).
        targets (torch.Tensor or np.ndarray): Ground truth values of shape (N, L, 5).
        scored_length (int): The number of sequence positions to score (default: 68).
        scored_cols_indices (list): Indices of the columns to include in the metric
                                    (default: [0, 1, 3] for reactivity, deg_Mg_pH10, deg_Mg_50C).

    Returns:
        float: The calculated MCRMSE.
    """
    if scored_cols_indices is None:
        scored_cols_indices = [0, 1, 3]

    # Convert Tensors to NumPy arrays if necessary
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Slice the data to the valid scoring region
    # Shape becomes: (N, scored_length, num_scored_cols)
    preds_scored = predictions[:, :scored_length, scored_cols_indices]
    targs_scored = targets[:, :scored_length, scored_cols_indices]

    # Calculate Squared Error
    squared_error = (preds_scored - targs_scored) ** 2

    # Calculate MSE per column (averaging over samples and sequence positions)
    # Axis 0 is Batch, Axis 1 is Sequence. Result shape: (num_scored_cols,)
    mse_per_col = np.mean(squared_error, axis=(0, 1))

    # Calculate RMSE per column
    rmse_per_col = np.sqrt(mse_per_col)

    # Calculate Mean Columnwise RMSE
    mcrmse = np.mean(rmse_per_col)

    return float(mcrmse)
