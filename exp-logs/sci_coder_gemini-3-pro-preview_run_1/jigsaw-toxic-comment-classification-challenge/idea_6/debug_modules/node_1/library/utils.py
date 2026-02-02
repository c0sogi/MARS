import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed: int = Config.seed):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.seed.
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


def get_score(y_true, y_pred):
    """
    Calculates the mean column-wise ROC AUC score.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth labels of shape (N, num_classes).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities of shape (N, num_classes).

    Returns:
        float: The mean column-wise ROC AUC.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Calculate ROC AUC
    # We use average='macro' to compute the metric for each label, and find their unweighted mean.
    # This satisfies the "Mean column-wise ROC AUC" metric requirement.
    try:
        score = roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        # This block handles edge cases, such as when a specific class has only one label
        # (all 0s or all 1s) in the provided batch/subset (common in debug mode).
        # In such cases, sklearn raises a ValueError. We calculate per-column manually.
        scores = []
        for i in range(y_true.shape[1]):
            try:
                # Only calculate if there are at least two classes present
                if len(np.unique(y_true[:, i])) > 1:
                    col_score = roc_auc_score(y_true[:, i], y_pred[:, i])
                    scores.append(col_score)
            except ValueError:
                continue

        if scores:
            score = np.mean(scores)
        else:
            # If no columns can be scored, return 0.5 (random guessing)
            score = 0.5

    return score
