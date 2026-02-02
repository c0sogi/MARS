import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mcrmse(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    The metric is calculated as the average of the RMSE values for each of the
    target columns.

    Args:
        y_true (np.array or torch.Tensor): Ground truth values.
                                           Shape: (N, num_targets) or (N, seq_len, num_targets).
        y_pred (np.array or torch.Tensor): Predicted values.
                                           Shape: must match y_true.

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

    # Flatten to (Total_Samples, Num_Targets) if input is 3D (Batch, Seq, Targets)
    # This handles the case where we evaluate over the sequence length
    if y_true.ndim == 3:
        # Reshape to (-1, num_targets) i.e., (-1, 3)
        num_targets = y_true.shape[-1]
        y_true = y_true.reshape(-1, num_targets)
        y_pred = y_pred.reshape(-1, num_targets)

    # Calculate MSE for each column (target)
    # axis=0 averages over the samples/positions
    mse_per_col = np.mean((y_true - y_pred) ** 2, axis=0)

    # Calculate RMSE for each column
    rmse_per_col = np.sqrt(mse_per_col)

    # Calculate the mean of the RMSEs
    score = np.mean(rmse_per_col)

    return score
