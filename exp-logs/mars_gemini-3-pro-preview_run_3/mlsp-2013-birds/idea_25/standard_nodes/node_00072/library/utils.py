import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Enforces deterministic behavior in cuDNN backends.

    Args:
        seed (int): The seed value to set. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_robust_auc(y_true, y_pred):
    """
    Computes the Macro-Averaged ROC AUC score, robustly handling classes with
    no positive samples or no negative samples in the provided batch/dataset.

    This prevents ValueError exceptions from sklearn when a batch is too small
    or a class is missing, which is common in multi-label classification on
    imbalanced datasets.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth labels of shape (N, num_classes).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities of shape (N, num_classes).

    Returns:
        float: The macro-averaged ROC AUC score. Returns 0.5 if no classes can be evaluated.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Handle empty inputs
    if y_true.size == 0 or y_pred.size == 0:
        return 0.5

    num_classes = y_true.shape[1]
    auc_scores = []

    for i in range(num_classes):
        true_col = y_true[:, i]
        pred_col = y_pred[:, i]

        # Check if the class has both positive and negative samples (0 and 1)
        # This is required for ROC AUC calculation
        if len(np.unique(true_col)) == 2:
            try:
                auc = roc_auc_score(true_col, pred_col)
                auc_scores.append(auc)
            except ValueError:
                # Fallback for unexpected sklearn errors
                pass
        else:
            # Skip classes that are constant in the ground truth for this batch
            # This effectively ignores them in the macro average
            pass

    if not auc_scores:
        # If no classes could be evaluated (e.g., all classes are constant in this batch)
        return 0.5

    return np.mean(auc_scores)
