import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_mcrmse(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    The metric is calculated by:
    1. Computing the RMSE for each target column separately over all samples and positions.
    2. Taking the average of these column-wise RMSE values.

    This handles the 'Mean of Sqrts' artifact correction described in the requirements.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth values.
                                             Expected shape: (N_samples, seq_len, num_targets)
                                             or (N_total, num_targets).
        y_pred (np.ndarray or torch.Tensor): Predicted values.
                                             Must have the same shape as y_true.

    Returns:
        float: The calculated MCRMSE score.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure shapes match
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch in metric calculation: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    # Calculate Squared Error: (y - y_hat)^2
    squared_error = np.square(y_true - y_pred)

    # Flatten spatial/batch dimensions to isolate the target columns
    # We assume the last dimension represents the different targets (channels)
    num_targets = y_true.shape[-1]
    squared_error_flat = squared_error.reshape(-1, num_targets)

    # Compute Mean Squared Error (MSE) for each column
    # Axis 0 represents all samples and sequence positions combined
    mse_per_column = np.mean(squared_error_flat, axis=0)

    # Compute Root Mean Squared Error (RMSE) for each column
    rmse_per_column = np.sqrt(mse_per_column)

    # Compute MCRMSE: Average of the column-wise RMSEs
    mcrmse = np.mean(rmse_per_column)

    return float(mcrmse)
