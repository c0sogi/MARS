import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import set_seed as config_set_seed


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Wraps the configuration's set_seed function to ensure consistent application.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    config_set_seed(seed)


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the macro-averaged Area Under the ROC Curve (ROC AUC).

    This function is designed to be robust for multi-label classification tasks
    where specific batches (e.g., during validation) might lack positive or
    negative samples for certain classes. It computes the AUC for each class
    independently and averages the results for all calculable classes.

    Args:
        y_true (np.array or torch.Tensor): Ground truth labels. Shape (N, Num_Classes).
        y_pred (np.array or torch.Tensor): Predicted probabilities. Shape (N, Num_Classes).

    Returns:
        float: The macro-averaged ROC AUC score. Returns 0.5 if no classes
               can be evaluated (e.g., constant labels across all samples).
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are at least 2D for consistent indexing
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    n_classes = y_true.shape[1]
    class_aucs = []

    for i in range(n_classes):
        # Extract columns for the current class
        y_true_cls = y_true[:, i]
        y_pred_cls = y_pred[:, i]

        # Check if the class has both positive and negative samples.
        # roc_auc_score raises ValueError if y_true has only one class.
        if len(np.unique(y_true_cls)) > 1:
            try:
                score = roc_auc_score(y_true_cls, y_pred_cls)
                class_aucs.append(score)
            except ValueError:
                # Fallback for any unexpected calculation errors
                pass

    # If no classes were valid for calculation, return a neutral score
    if not class_aucs:
        return 0.5

    return np.mean(class_aucs)
