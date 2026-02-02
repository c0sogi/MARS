import os
import random
import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef


def seed_everything(seed: int):
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
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_mcc(y_true, y_pred):
    """
    Calculates the Matthews Correlation Coefficient (MCC) between true labels and predictions.
    Handles both NumPy arrays and PyTorch tensors.

    Args:
        y_true (Union[np.ndarray, torch.Tensor, list]): Ground truth binary labels.
        y_pred (Union[np.ndarray, torch.Tensor, list]): Predicted binary labels.

    Returns:
        float: The Matthews Correlation Coefficient.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Convert lists to NumPy arrays
    if isinstance(y_true, list):
        y_true = np.array(y_true)

    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)

    # Ensure inputs are integers (binary labels)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    return matthews_corrcoef(y_true, y_pred)
