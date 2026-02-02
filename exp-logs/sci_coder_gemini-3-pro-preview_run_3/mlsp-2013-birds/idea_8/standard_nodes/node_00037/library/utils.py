import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed: int):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

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


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the macro-averaged ROC AUC score for multi-label classification.
    Handles cases where specific classes might be absent in the ground truth for a given batch.

    Args:
        y_true: Ground truth labels (N, NumClasses). Can be numpy array or torch tensor.
        y_pred: Predicted probabilities (N, NumClasses). Can be numpy array or torch tensor.

    Returns:
        float: The mean ROC AUC score across valid classes. Returns 0.0 if no classes are valid.
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
    aucs = []

    for i in range(n_classes):
        # Only calculate AUC if the class is present in the ground truth (both 0 and 1 exist)
        # This prevents ValueError: Only one class present in y_true.
        if len(np.unique(y_true[:, i])) > 1:
            try:
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                aucs.append(auc)
            except ValueError:
                # In case of any other unexpected error during calculation
                continue

    if not aucs:
        return 0.0

    return np.mean(aucs)
