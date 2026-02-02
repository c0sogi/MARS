import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_pos_weights(y_train, factor=1.0):
    """
    Calculates positive class weights for BCEWithLogitsLoss to handle class imbalance.
    Formula: weight = (negative_samples / positive_samples) * factor

    Args:
        y_train (np.ndarray): Binary label matrix of shape (n_samples, n_classes).
        factor (float): Scaling factor for the weights.

    Returns:
        torch.Tensor: Tensor of shape (n_classes,) containing the weights.
    """
    # Count positive samples per class
    pos_counts = np.sum(y_train, axis=0)
    total_samples = len(y_train)
    neg_counts = total_samples - pos_counts

    # Avoid division by zero by clamping positive counts to at least 1
    # This handles edge cases where a class might not appear in the training subset (unlikely but possible in debug)
    pos_counts = np.maximum(pos_counts, 1)

    # Calculate weights: negative / positive
    weights = (neg_counts / pos_counts) * factor

    return torch.tensor(weights, dtype=torch.float32)


def compute_auc(y_true, y_pred):
    """
    Computes the Macro-Averaged Area Under the ROC Curve.
    Handles cases where specific classes might be absent in the provided batch/set
    by iterating through columns if the global call fails.

    Args:
        y_true (np.ndarray): Ground truth binary labels (n_samples, n_classes).
        y_pred (np.ndarray): Predicted probabilities (n_samples, n_classes).

    Returns:
        float: The macro-averaged ROC AUC score.
    """
    try:
        # Try computing macro AUC directly using sklearn
        # This works if all classes are present in y_true
        return roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        # Fallback for cases where some classes have only one label (all 0s or all 1s) in y_true
        n_classes = y_true.shape[1]
        scores = []
        for i in range(n_classes):
            try:
                # Check if the class has both 0s and 1s
                if len(np.unique(y_true[:, i])) > 1:
                    s = roc_auc_score(y_true[:, i], y_pred[:, i])
                    scores.append(s)
            except ValueError:
                continue

        if len(scores) > 0:
            return np.mean(scores)
        else:
            # If no classes can be evaluated, return a neutral score
            return 0.5
