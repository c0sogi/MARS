import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import RANDOM_STATE


def set_seed(seed=RANDOM_STATE):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to RANDOM_STATE from config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_auc(y_true, y_pred):
    """
    Computes the Area Under the ROC Curve (AUC).

    Args:
        y_true (array-like): True binary labels.
        y_pred (array-like): Target scores or probability estimates.

    Returns:
        float: The ROC AUC score.
    """
    # Detach tensors if necessary and convert to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    try:
        score = roc_auc_score(y_true, y_pred)
        return score
    except ValueError as e:
        # This can happen if y_true has only one class
        print(f"Warning: Error computing AUC (likely single class in batch): {e}")
        return 0.5
