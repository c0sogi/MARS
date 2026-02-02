import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Enforce deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_roc_auc(y_true, y_pred):
    """
    Computes the Area Under the ROC Curve (ROC AUC) for multi-label classification.
    Calculates the metric for each class individually and returns the macro average.
    Handles cases where specific classes may be absent in the provided batch/set.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth labels (N, Num_Classes).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities (N, Num_Classes).

    Returns:
        float: The macro-averaged ROC AUC score.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are at least 2D
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    n_classes = y_true.shape[1]
    class_aucs = []

    for i in range(n_classes):
        # Check if the class exists in the ground truth (needs both 0s and 1s)
        if len(np.unique(y_true[:, i])) > 1:
            try:
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                class_aucs.append(auc)
            except ValueError:
                # Fallback for edge cases handled by unique check, but strictly safe
                continue
        else:
            # If a class is not present in the ground truth for this batch/split,
            # we skip it for the macro average calculation.
            continue

    if not class_aucs:
        return 0.0

    return np.mean(class_aucs)
