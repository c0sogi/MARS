import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The random seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the macro-averaged Area Under the ROC Curve (ROC AUC) for multi-label classification.
    Handles cases where specific classes might be absent in the ground truth for the current batch
    by skipping those classes in the average.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth binary labels of shape (N, num_classes).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities of shape (N, num_classes).

    Returns:
        float: The macro-averaged ROC AUC score.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Handle NaNs or Infs in predictions by replacing them with safe values
    if np.isnan(y_pred).any() or np.isinf(y_pred).any():
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1.0, neginf=0.0)

    # Calculate ROC AUC per class
    class_aucs = []
    num_classes = y_true.shape[1]

    for i in range(num_classes):
        # Check if the class is present in the ground truth (needs both 0s and 1s to compute AUC)
        # If a class has only one unique label in the batch (e.g. all 0s), AUC is undefined.
        if len(np.unique(y_true[:, i])) > 1:
            try:
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                class_aucs.append(auc)
            except ValueError:
                # Fallback if sklearn raises an error despite the unique check
                continue

    if len(class_aucs) == 0:
        # Return 0.5 (random guessing) if no classes can be evaluated in this batch
        return 0.5

    return np.mean(class_aucs)
