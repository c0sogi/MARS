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
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the macro-averaged Area Under the ROC Curve (ROC AUC) for multi-label predictions.
    Robustly handles cases where a specific class might not be present in the ground truth
    batch/fold by skipping that class in the average.

    Args:
        y_true (np.array or torch.Tensor): Ground truth labels of shape (N, num_classes).
        y_pred (np.array or torch.Tensor): Predicted probabilities of shape (N, num_classes).

    Returns:
        float: The macro-averaged ROC AUC score. Returns 0.5 if no classes are valid.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if hasattr(y_true, "detach"):
        y_true = y_true.detach().cpu().numpy()
    if hasattr(y_pred, "detach"):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    n_classes = y_true.shape[1]
    auc_scores = []

    for i in range(n_classes):
        # AUC can only be calculated if there are both positive and negative samples
        if len(np.unique(y_true[:, i])) == 2:
            try:
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
                auc_scores.append(score)
            except ValueError:
                # Fallback for edge cases not caught by unique check
                continue

    if len(auc_scores) == 0:
        return 0.5

    return np.mean(auc_scores)
