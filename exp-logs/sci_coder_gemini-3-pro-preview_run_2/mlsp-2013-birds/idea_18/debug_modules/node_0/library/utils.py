import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
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


def calculate_metrics(y_true, y_pred):
    """
    Calculates the Macro-Averaged Area Under the ROC Curve (ROC AUC).

    This function handles multi-label classification and is robust to cases
    where a specific class might not be present in the ground truth `y_true`
    (i.e., all zeros or all ones for a specific column). In such cases, that
    class is excluded from the average calculation to prevent errors.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth binary labels of shape (N, NumClasses).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities of shape (N, NumClasses).

    Returns:
        float: The macro-averaged ROC AUC score. Returns 0.0 if no classes can be evaluated.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    n_classes = y_true.shape[1]
    class_aucs = []

    for i in range(n_classes):
        # Check if the column contains both positive and negative samples
        # roc_auc_score is undefined if y_true contains only one class.
        if len(np.unique(y_true[:, i])) > 1:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            class_aucs.append(auc)

    # If no classes could be evaluated, return 0.0
    if not class_aucs:
        return 0.0

    # Return the mean of the valid AUCs (Macro Average)
    return np.mean(class_aucs)
