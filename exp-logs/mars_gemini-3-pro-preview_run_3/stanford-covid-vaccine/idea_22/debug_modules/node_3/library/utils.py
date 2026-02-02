import numpy as np
import torch
from library.config import Config


def set_seed(seed=None):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Delegates to the Config class implementation.

    Args:
        seed (int, optional): The seed value to set. If None, uses Config.SEED.
    """
    Config.set_seed(seed)


def compute_mcrmse(preds, targets, scored_indices=None):
    """
    Computes the Mean Columnwise Root Mean Squared Error (MCRMSE) for the specified columns.

    The competition scores 'reactivity', 'deg_Mg_pH10', and 'deg_Mg_50C'.
    Based on the submission format column order:
    0: reactivity
    1: deg_Mg_pH10
    2: deg_pH10
    3: deg_Mg_50C
    4: deg_50C

    Args:
        preds (np.ndarray or torch.Tensor): Predicted values. Shape (N, 5) or (N, Seq_Len, 5).
        targets (np.ndarray or torch.Tensor): Ground truth values. Shape must match preds.
        scored_indices (list, optional): Indices of columns to include in the metric.
                                         Defaults to [0, 1, 3] per competition rules.

    Returns:
        float: The MCRMSE score.
    """
    # Default to the competition scored columns if not provided
    if scored_indices is None:
        scored_indices = [0, 1, 3]

    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Ensure inputs are float64 for calculation precision
    preds = preds.astype(np.float64)
    targets = targets.astype(np.float64)

    # Reshape to 2D (Total_Samples, Num_Columns) to handle batch/sequence dimensions uniformly
    # If shape is (Batch, Seq, Cols), this flattens Batch and Seq into one dimension
    if preds.ndim == 3:
        preds = preds.reshape(-1, preds.shape[-1])
    if targets.ndim == 3:
        targets = targets.reshape(-1, targets.shape[-1])

    # Calculate Mean Squared Error for each column
    # axis=0 computes the mean across all samples/positions
    mse = np.mean((preds - targets) ** 2, axis=0)

    # Calculate Root Mean Squared Error for each column
    rmse = np.sqrt(mse)

    # Filter for the specific scored columns
    scored_rmse = rmse[scored_indices]

    # Compute the mean of the RMSEs
    mcrmse = np.mean(scored_rmse)

    return mcrmse
