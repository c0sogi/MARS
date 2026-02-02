import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Sets the random seed for various libraries to ensure reproducibility.

    Args:
        seed (int): The random seed to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_mcrmse(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    The metric is calculated by:
    1. Computing the RMSE for each target column separately (aggregating over samples and sequence positions).
    2. Taking the average of these column-wise RMSEs.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth values.
                                             Shape: (batch_size, seq_len, num_targets) or (N, num_targets).
        y_pred (np.ndarray or torch.Tensor): Predicted values.
                                             Shape: (batch_size, seq_len, num_targets) or (N, num_targets).

    Returns:
        float: The calculated MCRMSE value.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are float
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    # Flatten dimensions except the last one (targets)
    # If input is (Batch, Seq_Len, Targets), we want (Batch * Seq_Len, Targets)
    # This treats every position in every sample as an independent observation for that target column.
    if y_true.ndim == 3:
        num_targets = y_true.shape[-1]
        y_true = y_true.reshape(-1, num_targets)
        y_pred = y_pred.reshape(-1, num_targets)

    # Calculate MSE for each column
    # axis=0 aggregates over all samples/positions
    mse_per_col = np.mean((y_true - y_pred) ** 2, axis=0)

    # Calculate RMSE for each column
    rmse_per_col = np.sqrt(mse_per_col)

    # Calculate the mean of the RMSEs (MCRMSE)
    mcrmse = np.mean(rmse_per_col)

    return float(mcrmse)
