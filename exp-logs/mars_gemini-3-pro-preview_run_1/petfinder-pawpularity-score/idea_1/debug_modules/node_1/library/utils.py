import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Default is 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN to guarantee reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_rmse(y_true, y_pred):
    """
    Computes the Root Mean Squared Error (RMSE) between true and predicted values.

    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated target values.

    Returns:
        float: The RMSE value.
    """
    # Convert inputs to numpy arrays and flatten to ensure 1D comparison
    # This handles cases where inputs might be lists or have shape (N, 1)
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Calculate Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)

    # Return Root Mean Squared Error
    return np.sqrt(mse)
