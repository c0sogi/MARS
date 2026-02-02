import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import SEED


def set_seed(seed: int = SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # OS environment (hash seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic cuDNN behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve (AUC).

    Args:
        y_true (array-like or tensor): Ground truth binary labels.
        y_pred (array-like or tensor): Predicted probabilities for the positive class.

    Returns:
        float: The ROC AUC score.
    """
    # Handle PyTorch Tensors: detach from graph and convert to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    return roc_auc_score(y_true, y_pred)
