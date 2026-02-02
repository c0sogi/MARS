import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to set. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Returns the appropriate PyTorch device (CUDA or CPU) based on availability
    and configuration.

    Returns:
        torch.device: The device to be used for computation.
    """
    return torch.device(Config.DEVICE)


def mcrmse(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    MCRMSE = (1/Nt) * sum_j( sqrt( (1/n) * sum_i( (y_ij - y_hat_ij)^2 ) ) )

    Args:
        y_true (torch.Tensor or np.ndarray): Ground truth values.
        y_pred (torch.Tensor or np.ndarray): Predicted values.

    Returns:
        float: The MCRMSE score.
    """
    # Convert numpy arrays to torch tensors if necessary
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)

    # Ensure float32 for precision
    y_true = y_true.float()
    y_pred = y_pred.float()

    # Calculate MSE for each column (mean over samples/dim 0)
    mse = torch.mean((y_true - y_pred) ** 2, dim=0)

    # Calculate RMSE for each column
    rmse = torch.sqrt(mse)

    # Calculate the mean of the column-wise RMSEs
    score = torch.mean(rmse)

    return score.item()
