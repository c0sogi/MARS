import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the macro-averaged ROC AUC score for multi-label classification.
    Handles cases where specific classes may be missing from the batch/set.

    Args:
        y_true (np.ndarray): Ground truth binary labels of shape (N, num_classes).
        y_pred (np.ndarray): Predicted probabilities of shape (N, num_classes).

    Returns:
        float: Macro-averaged ROC AUC score.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    n_classes = y_true.shape[1]
    scores = []

    for i in range(n_classes):
        # Check if the class has both positive and negative samples in the ground truth
        # ROC AUC is undefined if only one class is present (all 0s or all 1s)
        if len(np.unique(y_true[:, i])) == 2:
            try:
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
                scores.append(score)
            except ValueError:
                # Skip columns where calculation fails for other reasons
                pass

    if not scores:
        # If no classes could be evaluated (e.g., extremely small batch with constant labels), return 0.5
        return 0.5

    return np.mean(scores)


def get_pos_weights(df_train, device):
    """
    Calculates class-specific positive weights for BCEWithLogitsLoss to handle class imbalance.
    Formula: pos_weight = number_of_negatives / number_of_positives

    Args:
        df_train (pd.DataFrame): Training dataframe containing 'species_X' columns.
        device (torch.device): The device to place the tensor on.

    Returns:
        torch.Tensor: A tensor of weights with shape (num_classes,).
    """
    # Identify label columns dynamically
    label_cols = [col for col in df_train.columns if col.startswith("species_")]

    # Sort columns by the integer suffix to ensure correct order (species_0, species_1, ..., species_10)
    # String sorting would place species_10 before species_2
    label_cols.sort(key=lambda x: int(x.split("_")[1]))

    if not label_cols:
        raise ValueError("No label columns found in dataframe (expected 'species_X').")

    # Calculate counts for each class
    pos_counts = df_train[label_cols].sum().values
    total_samples = len(df_train)
    neg_counts = total_samples - pos_counts

    # Calculate weights: N_neg / N_pos
    # Add a small epsilon to denominator to prevent division by zero
    weights = neg_counts / (pos_counts + 1e-6)

    # Convert to tensor and move to device
    pos_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    return pos_weights_tensor
