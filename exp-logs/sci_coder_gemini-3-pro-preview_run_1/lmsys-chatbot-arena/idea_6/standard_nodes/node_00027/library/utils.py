import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss
from library.config import Config


def seed_everything(seed=Config.SEED):
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

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Returns the appropriate PyTorch device (CUDA or CPU).

    Returns:
        torch.device: The device to use for computation.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(y_true, y_pred):
    """
    Computes the Log Loss metric for the competition.

    Args:
        y_true (np.array or torch.Tensor): Ground truth labels (probabilities or one-hot).
                                           Shape: (N, 3)
        y_pred (np.array or torch.Tensor): Predicted probabilities.
                                           Shape: (N, 3)

    Returns:
        float: The calculated log loss.
    """
    # Convert torch tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are float64 for precision
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    # Calculate Log Loss
    # Sklearn's log_loss handles clipping automatically (default eps=1e-15)
    # The metric is Log Loss with "eps=auto"
    score = log_loss(y_true, y_pred)

    return score
