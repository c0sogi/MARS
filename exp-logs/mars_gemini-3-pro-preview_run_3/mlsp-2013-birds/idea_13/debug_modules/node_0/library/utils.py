import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

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
        # Deterministic algorithms ensure reproducibility but might be slower
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_roc_auc(y_true, y_pred):
    """
    Computes the Macro-Averaged Area Under the ROC Curve.

    This function calculates the ROC AUC for each class individually and then computes
    the mean. It robustly handles cases where a specific class is not present (or
    only present) in the provided batch/subset by skipping that class in the
    average, preventing undefined metric errors.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth binary labels of shape (N, num_classes).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities of shape (N, num_classes).

    Returns:
        float: The macro-averaged ROC AUC score. Returns 0.0 if no classes are valid.
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
        # Extract the column for the current class
        y_true_col = y_true[:, i]
        y_pred_col = y_pred[:, i]

        # Check if the class has both positive and negative samples
        # sklearn.metrics.roc_auc_score requires both classes to be present
        if len(np.unique(y_true_col)) > 1:
            try:
                auc = roc_auc_score(y_true_col, y_pred_col)
                class_aucs.append(auc)
            except ValueError:
                # In case of any other sklearn error, skip this class
                continue
        else:
            # Skip classes that don't have both 0 and 1 in the ground truth
            continue

    if not class_aucs:
        return 0.0

    return np.mean(class_aucs)
