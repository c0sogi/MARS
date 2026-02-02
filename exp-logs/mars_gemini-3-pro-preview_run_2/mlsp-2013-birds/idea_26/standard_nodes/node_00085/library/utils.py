import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import CFG


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_pos_weights(df, device="cpu"):
    """
    Calculates positive weights for BCEWithLogitsLoss dynamically based on the provided DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the training data and label columns.
        device (str): Device to move the tensor to ('cpu' or 'cuda').

    Returns:
        torch.Tensor: Tensor of weights for each class.
    """
    # Identify label columns based on naming convention
    label_cols = [c for c in df.columns if c.startswith("species_")]

    # Validate that we found the expected number of classes
    if len(label_cols) != CFG.num_classes:
        raise ValueError(
            f"Expected {CFG.num_classes} label columns, found {len(label_cols)}"
        )

    # Calculate counts for each class
    counts = df[label_cols].sum().values
    total = len(df)

    # Calculate pos_weight = (Total - Count) / Count
    # Use np.maximum to prevent division by zero for completely absent classes (safety)
    weights = (total - counts) / np.maximum(counts, 1)

    return torch.tensor(weights, dtype=torch.float32).to(device)


def calculate_metric(y_true, y_pred):
    """
    Computes the Macro-Averaged Area Under the ROC Curve.
    Handles cases where a class might be constant in the current batch.

    Args:
        y_true (np.array or torch.Tensor): Ground truth labels (Binary).
        y_pred (np.array or torch.Tensor): Predicted probabilities or logits.

    Returns:
        float: Macro-averaged AUC score.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Calculate AUC per column to be robust against missing classes in small batches
    aucs = []
    num_classes = y_true.shape[1]

    for i in range(num_classes):
        try:
            # Check if the class has both 0s and 1s
            if len(np.unique(y_true[:, i])) > 1:
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                aucs.append(auc)
        except ValueError:
            # Skip columns where AUC is undefined
            continue

    if len(aucs) == 0:
        return 0.5  # Default fallback if no classes are valid

    return np.mean(aucs)
