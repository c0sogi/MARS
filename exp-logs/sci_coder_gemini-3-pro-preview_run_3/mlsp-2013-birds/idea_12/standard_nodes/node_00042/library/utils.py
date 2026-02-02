import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_robust_roc_auc(y_true, y_pred):
    """
    Computes the mean Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    averaged over all classes.

    This function is robust to batches where some classes may not be present (missing positive
    or negative samples). Such classes are excluded from the average for that specific batch.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth labels of shape (N, Num_Classes).
                                             Values should be binary (0 or 1).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities of shape (N, Num_Classes).
                                             Values should be float between 0 and 1.

    Returns:
        float: The mean ROC AUC score over valid classes. Returns 0.5 if no classes are valid.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
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
        class_true = y_true[:, i]
        class_pred = y_pred[:, i]

        # Check if the class has both positive and negative samples in this batch
        # np.unique returns sorted unique elements. If length is 2, we have both 0 and 1.
        if len(np.unique(class_true)) == 2:
            try:
                score = roc_auc_score(class_true, class_pred)
                auc_scores.append(score)
            except ValueError:
                # In extremely rare edge cases where roc_auc_score fails despite unique check
                continue

    # If no classes were valid (e.g., extremely small batch with constant labels), return neutral score
    if len(auc_scores) == 0:
        return 0.5

    return np.mean(auc_scores)
