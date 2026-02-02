import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_mcrmse(preds, targets):
    """
    Computes the Mean Columnwise Root Mean Squared Error (MCRMSE).

    The metric is calculated by:
    1. Computing the MSE for each target column separately over all samples and scored positions.
    2. Taking the square root to get RMSE per column.
    3. Taking the average of these column-wise RMSEs.

    This approach aligns with the strategy to average the RMSE of each column
    rather than averaging the RMSE of each position.

    Args:
        preds (np.ndarray or torch.Tensor): Predicted values.
                                            Expected shape: (N_samples, Seq_Len, N_columns)
        targets (np.ndarray or torch.Tensor): Ground truth values.
                                              Expected shape: (N_samples, Seq_Len, N_columns)

    Returns:
        float: The MCRMSE score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    preds = np.asarray(preds)
    targets = np.asarray(targets)

    # Check for shape consistency
    if preds.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch in compute_mcrmse: preds {preds.shape} vs targets {targets.shape}"
        )

    # Calculate MSE for each column (averaging over samples and sequence positions)
    # axis=0 is batch/samples, axis=1 is sequence length.
    # We want one scalar MSE per column (axis 2).
    # If inputs are flattened (N_total, N_cols), axis=0 handles it.
    # If inputs are (N, Seq, Cols), axis=(0, 1) handles it.
    reduce_axes = tuple(range(preds.ndim - 1))
    mse_per_col = np.mean((preds - targets) ** 2, axis=reduce_axes)

    # Calculate RMSE for each column
    rmse_per_col = np.sqrt(mse_per_col)

    # Calculate the mean of the column RMSEs
    mcrmse = np.mean(rmse_per_col)

    return float(mcrmse)
