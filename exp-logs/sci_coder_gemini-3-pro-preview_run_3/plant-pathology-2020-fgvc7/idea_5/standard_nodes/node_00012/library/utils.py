import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config, seed_everything as lib_seed_everything


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    Wraps the library implementation to ensure consistency.
    """
    lib_seed_everything(seed)


def get_class_weights(df: pd.DataFrame, label_cols: list = None) -> torch.Tensor:
    """
    Calculates inverse frequency weights for class balancing.
    Formula: w_j = n_samples / (n_classes * n_samples_j)

    Args:
        df (pd.DataFrame): The training dataframe containing the label columns.
        label_cols (list, optional): List of column names representing the classes.
                                     Defaults to Config.LABEL_COLS.

    Returns:
        torch.Tensor: A tensor of weights with shape (num_classes,) and dtype float32.
    """
    if label_cols is None:
        label_cols = Config.LABEL_COLS

    # Calculate the number of samples for each class
    # The dataset has one-hot encoded columns for the classes
    class_counts = df[label_cols].sum().values
    total_samples = len(df)
    num_classes = len(label_cols)

    # Avoid division by zero if a class has 0 samples (unlikely in this dataset)
    class_counts = np.clip(class_counts, a_min=1, a_max=None)

    # Compute balanced weights
    weights = total_samples / (num_classes * class_counts)

    return torch.tensor(weights, dtype=torch.float32)


def calculate_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Mean Column-wise ROC AUC score.

    Args:
        y_true (np.ndarray): Ground truth labels of shape (N, num_classes).
        y_pred (np.ndarray): Predicted probabilities of shape (N, num_classes).

    Returns:
        float: The mean ROC AUC score across all classes.
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    num_classes = y_true.shape[1]
    auc_scores = []

    for i in range(num_classes):
        try:
            # ROC AUC is only defined if there is at least one positive and one negative sample
            if len(np.unique(y_true[:, i])) > 1:
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
                auc_scores.append(score)
            else:
                # If a class is missing from the batch/fold, we cannot compute AUC for it.
                # In a stratified full validation set, this should not happen.
                pass
        except ValueError:
            pass

    if not auc_scores:
        return 0.0

    return np.mean(auc_scores)
