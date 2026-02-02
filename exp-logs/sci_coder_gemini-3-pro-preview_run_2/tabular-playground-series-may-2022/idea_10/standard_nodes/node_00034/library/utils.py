import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


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
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_auc(y_true, y_pred):
    """
    Computes the Area Under the ROC Curve (AUC).

    Args:
        y_true: Ground truth binary labels (numpy array or torch tensor).
        y_pred: Predicted probabilities (numpy array or torch tensor).

    Returns:
        float: The AUC score.
    """
    # Detach and move to cpu if they are torch tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are flat
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        # Fallback if y_true contains only one class (e.g. in very small batches)
        return 0.5
