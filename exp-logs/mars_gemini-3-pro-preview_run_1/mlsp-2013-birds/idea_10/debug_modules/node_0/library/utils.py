import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_roc_auc(y_true, y_pred, average="macro"):
    """
    Calculates the Area Under the Receiver Operating Characteristic Curve (ROC AUC).
    Handles cases where specific classes may not be present in the ground truth
    by calculating per-column AUC and averaging valid columns.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth labels (binary).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities.
        average (str): Averaging strategy ('macro', 'micro', 'weighted', 'samples').
                       Defaults to "macro".

    Returns:
        float: The computed ROC AUC score.
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    try:
        # Attempt standard calculation
        score = roc_auc_score(y_true, y_pred, average=average)
    except ValueError:
        # Fallback for cases where some classes have only one label (all 0s or all 1s)
        # This is common in small validation batches
        scores = []
        num_classes = y_true.shape[1]
        for i in range(num_classes):
            # Only calculate AUC if the class has both positive and negative samples
            if len(np.unique(y_true[:, i])) > 1:
                try:
                    s = roc_auc_score(y_true[:, i], y_pred[:, i])
                    scores.append(s)
                except ValueError:
                    pass

        if scores:
            score = np.mean(scores)
        else:
            # If no classes are valid, return a neutral score
            score = 0.5

    return float(score)


def sanitize_pseudo_labels(pseudo_labels):
    """
    Checks for NaN values in pseudo-labels and handles them by replacing with 0.0.
    Also ensures all probabilities are clipped within the [0, 1] range.

    Args:
        pseudo_labels (np.ndarray or torch.Tensor): The predicted probabilities.

    Returns:
        np.ndarray: Sanitized pseudo-labels.
    """
    if isinstance(pseudo_labels, torch.Tensor):
        pseudo_labels = pseudo_labels.detach().cpu().numpy()

    # Check for NaNs and replace
    if np.isnan(pseudo_labels).any():
        pseudo_labels = np.nan_to_num(pseudo_labels, nan=0.0)

    # Ensure probabilities are valid
    pseudo_labels = np.clip(pseudo_labels, 0.0, 1.0)

    return pseudo_labels
