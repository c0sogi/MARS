import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=Config.SEED):
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

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_roc_auc(y_true, y_pred):
    """
    Computes the Area Under the ROC Curve (ROC AUC) score.

    Args:
        y_true (array-like): True binary labels.
        y_pred (array-like): Target scores (probability estimates of the positive class).

    Returns:
        float: The computed ROC AUC score.
    """
    # sklearn.metrics.roc_auc_score handles numpy arrays and lists
    return roc_auc_score(y_true, y_pred)
