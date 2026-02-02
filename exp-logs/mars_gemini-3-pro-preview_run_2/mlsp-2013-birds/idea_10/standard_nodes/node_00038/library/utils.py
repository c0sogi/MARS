import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets fixed random seeds for reproducibility across Python, NumPy, and PyTorch.
    Ensures deterministic behavior for CUDA operations.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the macro-averaged Area Under the ROC Curve (ROC AUC).
    Manually iterates over classes to handle sparse data where some classes
    may be missing in the validation fold.

    Args:
        y_true (np.ndarray): Ground truth binary labels (N_samples, N_classes).
        y_pred (np.ndarray): Predicted probabilities (N_samples, N_classes).

    Returns:
        float: The macro-averaged ROC AUC score. Returns 0.5 if calculation fails completely.
    """
    scores = []
    num_classes = y_true.shape[1]

    # Iterate manually to avoid sklearn returning NaN for 'macro' average on sparse data
    # (Cite debug_lesson_6)
    for i in range(num_classes):
        # Only calculate if both classes are present (0 and 1)
        if len(np.unique(y_true[:, i])) > 1:
            try:
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
                scores.append(score)
            except ValueError:
                continue

    if len(scores) > 0:
        return np.mean(scores)
    else:
        return 0.5


def mixup_data(x, y, alpha=Config.MIXUP_ALPHA, device=Config.DEVICE):
    """
    Applies Mixup augmentation to the input batch.
    Samples lambda from Beta(alpha, alpha).

    Args:
        x (torch.Tensor): Input images batch.
        y (torch.Tensor): Input labels batch.
        alpha (float): Mixup interpolation coefficient.
        device (str): Device to perform operations on.

    Returns:
        mixed_x (torch.Tensor): Mixed images.
        y_a (torch.Tensor): Labels for the first permutation.
        y_b (torch.Tensor): Labels for the second permutation.
        lam (float): Lambda mixing coefficient.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def calculate_pos_weights(train_df, device=Config.DEVICE):
    """
    Calculates positive class weights for BCEWithLogitsLoss to handle class imbalance.
    Weight = Count(Negative) / Count(Positive).

    Args:
        train_df (pd.DataFrame): DataFrame containing training metadata and labels.
        device (str): Device to place the weights tensor on.

    Returns:
        torch.Tensor: Tensor of weights for each class (shape: [NUM_CLASSES]).
    """
    # Identify label columns based on configuration
    label_cols = [f"species_{i}" for i in range(Config.NUM_CLASSES)]

    # Extract label matrix
    labels = train_df[label_cols].values

    # Calculate counts
    pos_counts = np.sum(labels, axis=0)
    total_counts = len(labels)
    neg_counts = total_counts - pos_counts

    # Calculate weights: neg / pos
    # Add epsilon to avoid division by zero
    weights = neg_counts / (pos_counts + 1e-6)

    return torch.tensor(weights, dtype=torch.float32).to(device)
