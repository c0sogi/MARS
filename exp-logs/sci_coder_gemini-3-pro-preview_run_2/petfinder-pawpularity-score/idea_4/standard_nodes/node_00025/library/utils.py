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

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def scale_target(targets):
    """
    Scales the Pawpularity score from [1, 100] to [0, 1] for Sigmoid activation.

    Args:
        targets (float, np.ndarray, or torch.Tensor): The original targets.

    Returns:
        The scaled targets.
    """
    return targets / 100.0


def unscale_target(targets):
    """
    Unscales the prediction from [0, 1] back to the original [1, 100] range.

    Args:
        targets (float, np.ndarray, or torch.Tensor): The scaled targets.

    Returns:
        The unscaled targets.
    """
    return targets * 100.0


def get_rmse(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error (RMSE) between true and predicted values.

    Args:
        y_true (list, np.ndarray, torch.Tensor): The ground truth values.
        y_pred (list, np.ndarray, torch.Tensor): The predicted values.

    Returns:
        float: The RMSE value.
    """
    # Handle torch tensors by detaching and moving to cpu
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Convert to numpy arrays if they are lists
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)
