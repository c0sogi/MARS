import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metric(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve (AUC) for validation.

    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred (array-like): Predicted probabilities.

    Returns:
        float: The ROC AUC score.
    """
    # Ensure inputs are numpy arrays for compatibility with sklearn
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return roc_auc_score(y_true, y_pred)
