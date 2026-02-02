import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Default is 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_multilabel_auc(y_true, y_pred):
    """
    Computes the Area Under the ROC Curve (AUC) for multi-label classification.
    Explicitly handles edge cases where specific classes are absent in the validation batch
    to prevent metric calculation errors.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth labels of shape (N_samples, N_classes).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities of shape (N_samples, N_classes).

    Returns:
        float: The macro-averaged AUC score across all classes present in the batch.
               Returns 0.0 if no classes can be evaluated.
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
        # Check if the class has both positive and negative samples in this batch
        # roc_auc_score requires both classes to be present
        if len(np.unique(y_true[:, i])) == 2:
            try:
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
                auc_scores.append(score)
            except ValueError:
                # In case of any other unexpected error from sklearn
                continue

    if len(auc_scores) == 0:
        return 0.0

    return np.mean(auc_scores)
