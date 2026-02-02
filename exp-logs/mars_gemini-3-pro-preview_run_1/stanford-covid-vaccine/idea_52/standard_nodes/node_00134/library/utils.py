import os
import random
import numpy as np
import torch
from library.config import Config


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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Determines the available hardware device for computation.

    Returns:
        torch.device: Returns 'cuda' if a GPU is available, otherwise 'cpu'.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mcrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    The metric is calculated as the average of the RMSE values for each of the
    scored target columns.

    Formula:
    MCRMSE = (1/Nt) * sum_j( sqrt( (1/n) * sum_i( (y_ij - y_hat_ij)^2 ) ) )

    Args:
        y_true (np.ndarray): Ground truth values. Shape should be (N, ..., num_columns).
                             Expected to contain only the scored columns.
        y_pred (np.ndarray): Predicted values. Shape must match y_true.

    Returns:
        float: The MCRMSE score.
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    assert (
        y_true.shape == y_pred.shape
    ), f"Shape mismatch: {y_true.shape} vs {y_pred.shape}"

    # Calculate squared errors
    squared_errors = (y_true - y_pred) ** 2

    # Calculate Mean Squared Error for each column (averaging over all other dimensions)
    # We assume the last dimension represents the target columns (channels)
    # reshape to (-1, num_columns) to flatten batch and sequence dimensions
    num_columns = y_true.shape[-1]
    y_true_flat = y_true.reshape(-1, num_columns)
    y_pred_flat = y_pred.reshape(-1, num_columns)

    mse_per_column = np.mean((y_true_flat - y_pred_flat) ** 2, axis=0)

    # Calculate RMSE for each column
    rmse_per_column = np.sqrt(mse_per_column)

    # Calculate the mean of the RMSEs
    score = np.mean(rmse_per_column)

    return float(score)
