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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the mean column-wise ROC AUC score.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth binary labels (N, num_classes).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities (N, num_classes).

    Returns:
        float: The mean ROC AUC score across all columns.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    n_cols = y_true.shape[1]
    auc_scores = []

    for i in range(n_cols):
        # Check if the column has both classes (0 and 1)
        # ROC AUC is undefined if only one class is present in y_true
        if len(np.unique(y_true[:, i])) == 2:
            score = roc_auc_score(y_true[:, i], y_pred[:, i])
            auc_scores.append(score)
        else:
            # If a class is missing in the ground truth for this batch/split,
            # we cannot calculate AUC for this column.
            # In a proper stratified validation set, this should not happen often.
            pass

    if not auc_scores:
        return 0.0

    return np.mean(auc_scores)
