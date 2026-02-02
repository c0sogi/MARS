import os
import random
import numpy as np
import torch
from library.config import SEED


def seed_everything(seed: int = SEED) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_mcrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    This metric computes the RMSE for each column independently and then
    takes the average of these RMSE values. This avoids the 'Mean of Sqrts'
    artifact that can occur if averaging MSEs first.

    Args:
        y_true (np.ndarray): Ground truth values array of shape (N, 3).
                             Columns should correspond to the scored targets
                             (reactivity, deg_Mg_pH10, deg_Mg_50C).
        y_pred (np.ndarray): Predicted values array of shape (N, 3).
                             Columns should correspond to the scored targets.

    Returns:
        float: The calculated MCRMSE score.
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Check for shape consistency
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    # Calculate Mean Squared Error (MSE) for each column
    # axis=0 averages over the samples (rows)
    colwise_mse = np.mean(np.square(y_true - y_pred), axis=0)

    # Calculate Root Mean Squared Error (RMSE) for each column
    colwise_rmse = np.sqrt(colwise_mse)

    # Calculate the mean of the column RMSEs (MCRMSE)
    mcrmse_score = np.mean(colwise_rmse)

    return float(mcrmse_score)
