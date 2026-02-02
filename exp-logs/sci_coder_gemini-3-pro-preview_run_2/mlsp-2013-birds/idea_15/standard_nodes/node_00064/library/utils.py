import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import set_seed


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Wraps the implementation provided in library.config.

    Args:
        seed (int): The seed value to use.
    """
    set_seed(seed)


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the macro-averaged Area Under the ROC Curve (AUC) for multi-label classification.

    This function handles cases where a specific class might not have both positive and negative
    samples in the provided batch/dataset by skipping that class in the average calculation,
    preventing ValueError from sklearn.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth labels of shape (N_samples, N_classes).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities of shape (N_samples, N_classes).

    Returns:
        float: The macro-averaged ROC AUC score. Returns 0.5 if AUC cannot be calculated for any class.
    """
    # Convert PyTorch tensors to NumPy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    n_classes = y_true.shape[1]
    auc_scores = []

    for i in range(n_classes):
        # Extract columns for the current class
        y_true_col = y_true[:, i]
        y_pred_col = y_pred[:, i]

        # Check if both classes (0 and 1) are present
        if len(np.unique(y_true_col)) > 1:
            try:
                score = roc_auc_score(y_true_col, y_pred_col)
                auc_scores.append(score)
            except ValueError:
                # Fallback for unexpected errors in calculation
                continue

    if not auc_scores:
        return 0.5

    return np.mean(auc_scores)
