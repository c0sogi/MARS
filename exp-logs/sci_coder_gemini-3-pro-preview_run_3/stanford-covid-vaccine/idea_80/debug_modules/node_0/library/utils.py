import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Selects the appropriate device (GPU if available, else CPU).

    Returns:
        torch.device: The device object.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def mcrmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    Metric definition:
    MCRMSE = (1 / Nt) * sum_j( sqrt( (1 / n) * sum_i( (y_ij - y_hat_ij)^2 ) ) )

    Args:
        y_true (torch.Tensor): Ground truth values of shape (batch_size, num_targets).
        y_pred (torch.Tensor): Predicted values of shape (batch_size, num_targets).

    Returns:
        torch.Tensor: A scalar tensor containing the MCRMSE score.
    """
    # Calculate MSE for each column (target)
    # shape: (num_targets,)
    col_mse = torch.mean((y_true - y_pred) ** 2, dim=0)

    # Calculate RMSE for each column
    col_rmse = torch.sqrt(col_mse)

    # Calculate the mean of the column RMSEs
    return torch.mean(col_rmse)
