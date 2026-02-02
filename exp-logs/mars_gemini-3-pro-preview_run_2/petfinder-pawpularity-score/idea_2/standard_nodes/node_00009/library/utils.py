import os
import random
import numpy as np
import torch
import math
from sklearn.metrics import mean_squared_error


def seed_everything(seed: int = 42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_rmse_score(outputs, targets):
    """
    Calculates the Root Mean Squared Error (RMSE) between predictions and targets.
    Handles both PyTorch tensors and NumPy arrays.

    Args:
        outputs (torch.Tensor or np.ndarray): The predicted values.
        targets (torch.Tensor or np.ndarray): The ground truth values.

    Returns:
        float: The RMSE score.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    outputs = np.array(outputs)
    targets = np.array(targets)

    # Calculate MSE
    mse = mean_squared_error(targets, outputs)

    # Return RMSE
    return math.sqrt(mse)
