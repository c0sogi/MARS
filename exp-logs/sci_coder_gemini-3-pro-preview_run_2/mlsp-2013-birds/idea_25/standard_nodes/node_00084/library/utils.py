import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Ensures deterministic behavior for cuDNN backend.

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

    # Enforce deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_multilabel_auc(y_true, y_pred):
    """
    Calculates the macro-averaged ROC AUC for multi-label classification.
    Robustly handles cases where specific classes may not be present in the
    ground truth (e.g., small batches or rare species) by skipping them
    in the average calculation.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth labels of shape (N, Num_Classes).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities of shape (N, Num_Classes).

    Returns:
        float: The macro-averaged ROC AUC score. Returns 0.0 if no classes are valid.
    """
    # Convert tensors to numpy if necessary
    if hasattr(y_true, "cpu"):
        y_true = y_true.detach().cpu().numpy()
    if hasattr(y_pred, "cpu"):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    n_classes = y_true.shape[1]
    auc_scores = []

    for i in range(n_classes):
        # ROC AUC requires both positive and negative samples
        # We check if the current class column has more than 1 unique value
        if len(np.unique(y_true[:, i])) > 1:
            try:
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
                auc_scores.append(score)
            except ValueError:
                # Fallback for edge cases not caught by unique check
                continue

    if not auc_scores:
        return 0.0

    return np.mean(auc_scores)
