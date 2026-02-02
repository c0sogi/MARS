import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed: int = Config.SEED) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

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


def compute_auc(y_true, y_pred) -> float:
    """
    Calculates the Area Under the ROC Curve (AUC).

    Args:
        y_true: Ground truth binary labels (0 or 1). Can be list, numpy array, or torch tensor.
        y_pred: Predicted probabilities. Can be list, numpy array, or torch tensor.

    Returns:
        float: The ROC AUC score.
    """
    # Convert torch tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate AUC
    # Note: roc_auc_score requires both classes to be present in y_true.
    # If called on a small batch with only one class, this might raise a ValueError.
    # We assume this is called on a sufficiently large set (e.g., validation epoch end).
    return roc_auc_score(y_true, y_pred)
