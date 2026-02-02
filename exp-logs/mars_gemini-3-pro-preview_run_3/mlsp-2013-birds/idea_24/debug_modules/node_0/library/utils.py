import os
import random
import numpy as np
import torch
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the mean Area Under the ROC Curve for multi-label data.
    Robustly handles cases where a batch or fold might lack positive or negative
    samples for specific classes by masking them out to prevent NaN scores.

    Args:
        y_true: Ground truth labels (numpy array or torch tensor), shape (N, num_classes).
        y_pred: Predicted probabilities (numpy array or torch tensor), shape (N, num_classes).

    Returns:
        float: Mean ROC AUC score across valid classes. Returns 0.5 if no classes are valid.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    num_classes = y_true.shape[1]
    auc_scores = []

    for i in range(num_classes):
        # Extract columns for the current class
        class_true = y_true[:, i]
        class_pred = y_pred[:, i]

        # Check if the class has both positive and negative samples (at least one 0 and one 1)
        # If a class is all 0s or all 1s in this batch, AUC is undefined for that class.
        if len(np.unique(class_true)) > 1:
            try:
                score = roc_auc_score(class_true, class_pred)
                auc_scores.append(score)
            except ValueError:
                # In case of any other calculation error, skip this class
                continue

    # If no classes were valid (e.g., extremely small batch with constant labels), return neutral score
    if not auc_scores:
        return 0.5

    return np.mean(auc_scores)
