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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_mcc(y_true, y_pred, threshold=0.5):
    """
    Calculates the Matthews Correlation Coefficient.

    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred (array-like): Predicted labels or probabilities.
        threshold (float): Threshold to convert probabilities to binary labels.
                           Defaults to 0.5.

    Returns:
        float: The MCC score.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # If predictions are probabilities (floating point), apply threshold
    if np.issubdtype(y_pred.dtype, np.floating):
        y_pred = (y_pred >= threshold).astype(int)

    return matthews_corrcoef(y_true, y_pred)
