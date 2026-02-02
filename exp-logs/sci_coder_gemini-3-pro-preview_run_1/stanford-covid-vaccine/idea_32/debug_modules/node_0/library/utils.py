import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
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
    torch.cuda.manual_seed_all(seed)

    # Enforce deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metric_mcrmse(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    This metric computes the RMSE for each target column separately and then
    averages them. This is the official scoring metric for the competition.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth values.
            Shape should be (N, num_targets) or (N, seq_len, num_targets).
        y_pred (np.ndarray or torch.Tensor): Predicted values.
            Shape must match y_true.

    Returns:
        float: The MCRMSE score.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure shapes match
    assert (
        y_true.shape == y_pred.shape
    ), f"Shape mismatch: {y_true.shape} vs {y_pred.shape}"

    # Calculate MSE for each column (target)
    # If the input is 3D (Batch, Seq, Channels), we flatten the first two dimensions
    # effectively treating every position as a sample for the column-wise calculation.
    # Alternatively, we can just take the mean over all axes except the last one (channels).

    # We calculate the mean squared error across all samples for each column (last axis)
    # axis=tuple(range(y_true.ndim - 1)) selects all dimensions except the last one (channels)
    mse_per_column = np.mean((y_true - y_pred) ** 2, axis=tuple(range(y_true.ndim - 1)))

    # Calculate RMSE for each column
    rmse_per_column = np.sqrt(mse_per_column)

    # Calculate the mean of the column-wise RMSEs
    mcrmse = np.mean(rmse_per_column)

    return float(mcrmse)
