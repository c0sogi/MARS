import os
import random
import numpy as np
import torch
from scipy.stats import pearsonr


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(preds, labels):
    """
    Computes the Pearson correlation coefficient between predictions and labels.

    Args:
        preds (array-like): Predicted similarity scores.
        labels (array-like): Ground truth similarity scores.

    Returns:
        float: The Pearson correlation coefficient.
    """
    # Ensure inputs are numpy arrays and flattened to 1D
    preds_flat = np.array(preds).reshape(-1)
    labels_flat = np.array(labels).reshape(-1)

    # Calculate Pearson correlation
    # pearsonr returns (statistic, p-value), we only need the statistic
    correlation, _ = pearsonr(preds_flat, labels_flat)

    return correlation
