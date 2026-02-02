import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed: int = Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU.

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_pos_weights(
    df: pd.DataFrame, device: torch.device = Config.DEVICE
) -> torch.Tensor:
    """
    Calculates positive weights for BCEWithLogitsLoss based on class frequencies.
    Formula: pos_weight = negative_count / positive_count

    Args:
        df (pd.DataFrame): The training metadata DataFrame containing label columns.
        device (torch.device): The device to move the weights tensor to.

    Returns:
        torch.Tensor: A tensor of shape (num_classes,) containing the weights.
    """
    # Identify label columns based on the 'species_' prefix
    label_cols = [col for col in df.columns if col.startswith("species_")]

    # Extract the label matrix
    labels = df[label_cols].values

    # Count positives and negatives for each class
    pos_counts = np.sum(labels, axis=0)
    neg_counts = len(df) - pos_counts

    # Calculate weights: neg / pos
    # Add a small epsilon to pos_counts to avoid division by zero if a class is empty
    weights = neg_counts / (pos_counts + 1e-6)

    # Convert to tensor
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    return weights_tensor


def compute_auc(y_true, y_pred) -> float:
    """
    Computes the Macro-Averaged Area Under the ROC Curve.
    Handles both numpy arrays and torch tensors.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth labels (N, num_classes).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities (N, num_classes).

    Returns:
        float: The macro-averaged ROC AUC score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Calculate Macro AUC
    # We use a try-except block or column-wise calculation to be robust
    # against cases where a class might have only one label in the provided batch/subset.
    try:
        score = roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        score = np.nan

    # Cite debug_lesson_3: Sanitize Metrics. Fallback if macro average returns NaN (sparse data).
    if np.isnan(score):
        # Fallback: Calculate per column and ignore columns with only one class present
        aucs = []
        for i in range(y_true.shape[1]):
            try:
                if len(np.unique(y_true[:, i])) > 1:
                    aucs.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
            except ValueError:
                pass

        if len(aucs) > 0:
            score = np.mean(aucs)
        else:
            score = 0.5  # Default random guess if undefined

    return float(score)
