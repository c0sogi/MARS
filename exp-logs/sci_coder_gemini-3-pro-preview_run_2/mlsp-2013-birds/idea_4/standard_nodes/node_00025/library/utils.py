import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_metric(y_true, y_pred):
    """
    Calculates the Macro-Averaged ROC AUC score.
    Handles cases where specific classes might be absent in the ground truth
    by skipping them in the average.

    Args:
        y_true (np.array or torch.Tensor): Ground truth labels (N, NumClasses).
        y_pred (np.array or torch.Tensor): Predicted probabilities (N, NumClasses).

    Returns:
        float: The macro-averaged ROC AUC score.
    """
    # Convert tensors to numpy if they are tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Calculate AUC per class to handle potential errors with missing classes
    # in small validation batches (e.g., a class has all 0s).
    n_classes = y_true.shape[1]
    class_aucs = []

    for i in range(n_classes):
        # Only calculate AUC if the class has both positive and negative samples
        # roc_auc_score throws ValueError if y_true has only one class
        if len(np.unique(y_true[:, i])) > 1:
            try:
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                class_aucs.append(auc)
            except ValueError:
                # Fallback if something unexpected happens
                pass

    if not class_aucs:
        return 0.5  # Default if no classes can be evaluated

    return np.mean(class_aucs)
