import os
import random
import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef
from library.config import SEED, DEVICE


def seed_everything(seed: int = SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

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
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_mcc(y_true, y_pred):
    """
    Calculates the Matthews Correlation Coefficient (MCC).

    Args:
        y_true: Ground truth binary labels (numpy array or torch tensor).
        y_pred: Predicted binary labels (numpy array or torch tensor).

    Returns:
        float: The MCC score.
    """
    # Convert torch tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are integer arrays (binary 0/1)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    return matthews_corrcoef(y_true, y_pred)


def get_device():
    """
    Returns the PyTorch device to be used for computation.

    Returns:
        torch.device: The device object (cuda or cpu).
    """
    return torch.device(DEVICE)
