import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed: int = Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash seeding
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """
    Returns the appropriate PyTorch device based on availability.

    Returns:
        torch.device: 'cuda' if available, otherwise 'cpu'.
    """
    return torch.device(Config.DEVICE)


def mcrmse_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE)
    for the specific scored columns defined in the competition metric.

    Formula: MCRMSE = (1/Nt) * sum_j( sqrt( (1/n) * sum_i( (y_ij - y_hat_ij)^2 ) ) )

    Args:
        y_true (np.ndarray): Ground truth values of shape (N, 5).
        y_pred (np.ndarray): Predicted values of shape (N, 5).

    Returns:
        float: The calculated MCRMSE score.
    """
    # Determine the indices of the columns that contribute to the score
    # Config.TARGET_COLS contains all 5 columns
    # Config.SCORED_COLS contains the 3 columns used for scoring
    scored_indices = [
        i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
    ]

    # Filter the arrays to include only the scored columns
    y_true_scored = y_true[:, scored_indices]
    y_pred_scored = y_pred[:, scored_indices]

    # Calculate Mean Squared Error for each column (axis 0 is the sample dimension)
    mse_per_column = np.mean((y_true_scored - y_pred_scored) ** 2, axis=0)

    # Calculate Root Mean Squared Error for each column
    rmse_per_column = np.sqrt(mse_per_column)

    # Calculate the mean of the RMSEs across the scored columns
    mcrmse = np.mean(rmse_per_column)

    return float(mcrmse)
