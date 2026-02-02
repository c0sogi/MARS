import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # PyTorch seeding
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in PyTorch backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_roc_auc(y_true, y_score):
    """
    Computes the Area Under the Receiver Operating Characteristic Curve (ROC AUC).

    Args:
        y_true (array-like): True binary labels.
        y_score (array-like): Target scores (probability estimates of the positive class).

    Returns:
        float: The ROC AUC score.
    """
    return roc_auc_score(y_true, y_score)
