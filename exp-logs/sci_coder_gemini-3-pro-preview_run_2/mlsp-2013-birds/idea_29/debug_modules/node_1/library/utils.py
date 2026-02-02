import os
import random
import numpy as np
import torch
import pandas as pd
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


def get_pos_weights(df, label_cols, device="cpu"):
    """
    Calculates class balancing weights for BCEWithLogitsLoss based on the ratio
    of negative to positive samples for each class.

    Formula: pos_weight = number_of_negatives / number_of_positives

    Args:
        df (pd.DataFrame): DataFrame containing the training labels.
        label_cols (list): List of column names corresponding to the targets.
        device (str or torch.device): Device to move the weights tensor to.

    Returns:
        torch.Tensor: Tensor of weights with shape (num_classes,).
    """
    # Extract label matrix
    labels = df[label_cols].values

    # Count positives and negatives for each class
    num_pos = np.sum(labels, axis=0)
    num_neg = len(labels) - num_pos

    # Calculate weights (add epsilon to avoid division by zero)
    weights = num_neg / (num_pos + 1e-6)

    # Convert to tensor and move to specified device
    pos_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    return pos_weights


def compute_auc(y_true, y_pred):
    """
    Computes the Area Under the ROC Curve (AUC) using macro averaging.

    Args:
        y_true (np.array or torch.Tensor): Ground truth labels (multi-hot).
        y_pred (np.array or torch.Tensor): Predicted probabilities.

    Returns:
        float: Macro-averaged ROC AUC score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Handle NaNs in predictions if any
    if np.isnan(y_pred).any():
        y_pred = np.nan_to_num(y_pred)

    try:
        # Calculate Macro AUC
        # This computes AUC for each class and averages them
        auc_score = roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        # Fallback for cases where a class might not be present in y_true
        # Calculate per class manually and ignore invalid classes
        aucs = []
        for i in range(y_true.shape[1]):
            try:
                # Only calculate if both classes are present
                if len(np.unique(y_true[:, i])) > 1:
                    score = roc_auc_score(y_true[:, i], y_pred[:, i])
                    aucs.append(score)
            except ValueError:
                continue

        if len(aucs) > 0:
            auc_score = np.mean(aucs)
        else:
            # Default fallback if calculation is impossible
            auc_score = 0.5

    return float(auc_score)
