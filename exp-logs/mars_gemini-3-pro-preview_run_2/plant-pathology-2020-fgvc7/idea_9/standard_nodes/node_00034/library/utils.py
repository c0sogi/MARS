import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score


def seed_everything(seed=42):
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


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the mean column-wise ROC AUC score.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth labels of shape (n_samples, n_classes).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities of shape (n_samples, n_classes).

    Returns:
        float: The mean column-wise ROC AUC score.
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Check if we have valid data for AUC calculation
    # roc_auc_score requires at least one positive and one negative sample per class
    # to be well-defined.
    try:
        score = roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        # In case of errors (e.g. only one class present in the batch), return a neutral score
        score = 0.5

    return score


def get_class_weights(df, target_cols):
    """
    Calculates the positive class weights (inverse frequency) for binary targets.
    Used for the pos_weight argument in BCEWithLogitsLoss.
    Formula: weight = number_of_negatives / number_of_positives

    Args:
        df (pd.DataFrame): The dataframe containing the target columns.
        target_cols (list of str): The names of the target columns.

    Returns:
        torch.Tensor: A tensor of weights with shape (len(target_cols),).
    """
    weights = []
    for col in target_cols:
        if col not in df.columns:
            raise ValueError(f"Target column '{col}' not found in DataFrame.")

        # Count positives and negatives
        pos_count = df[col].sum()
        total_count = len(df)
        neg_count = total_count - pos_count

        # Calculate weight
        if pos_count == 0:
            # Handle edge case where a class has no positive samples
            weight = 1.0
        else:
            weight = neg_count / pos_count

        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)
