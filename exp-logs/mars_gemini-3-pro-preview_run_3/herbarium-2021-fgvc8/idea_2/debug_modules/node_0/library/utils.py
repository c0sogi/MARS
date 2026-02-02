import numpy as np
import torch
from sklearn.metrics import f1_score
from library.config import seed_everything


def calculate_metric(y_true, y_pred):
    """
    Calculates the Macro F1 score for the given ground truth and predictions.

    This function handles both PyTorch tensors and NumPy arrays/lists as input.

    Args:
        y_true (array-like or torch.Tensor): Ground truth (correct) target values.
        y_pred (array-like or torch.Tensor): Estimated targets as returned by a classifier.

    Returns:
        float: Macro F1 score.
    """
    # Handle PyTorch Tensors: detach from graph and move to CPU
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate Macro F1 score
    # The metric requires a separate F1 score for each species value, averaged.
    return f1_score(y_true, y_pred, average="macro")
