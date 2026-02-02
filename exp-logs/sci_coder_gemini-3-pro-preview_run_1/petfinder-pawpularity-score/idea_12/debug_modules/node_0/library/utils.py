import os
import random
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_rmse(y_true, y_pred):
    """
    Computes the Root Mean Squared Error (RMSE) between true and predicted values.

    Args:
        y_true (array-like or torch.Tensor): Ground truth target values.
        y_pred (array-like or torch.Tensor): Estimated target values.

    Returns:
        float: The RMSE value.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)
