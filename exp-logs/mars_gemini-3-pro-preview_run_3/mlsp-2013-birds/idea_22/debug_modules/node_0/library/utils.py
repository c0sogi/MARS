import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed: int = 42):
    """
    Sets fixed random seeds for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The random seed to use.
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


def get_device() -> torch.device:
    """
    Selects the available hardware device for training.

    Returns:
        torch.device: 'cuda' if available, otherwise 'cpu'.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_roc_auc(y_true, y_pred) -> float:
    """
    Calculates the macro-averaged ROC AUC score in a robust manner.
    It handles cases where specific classes might be absent (all 0s or all 1s)
    in a given batch, which would otherwise cause sklearn to raise an error.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth labels of shape (N, num_classes).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities of shape (N, num_classes).

    Returns:
        float: The macro-averaged ROC AUC score over valid classes.
               Returns 0.5 if no classes are valid in the batch.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    n_classes = y_true.shape[1]
    auc_scores = []

    for i in range(n_classes):
        # ROC AUC is only defined if there are both positive and negative samples
        # for the specific class in the current batch.
        if len(np.unique(y_true[:, i])) == 2:
            try:
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
                auc_scores.append(score)
            except ValueError:
                # In case of any other sklearn specific edge cases
                continue

    if not auc_scores:
        # If no classes were valid (e.g., extremely small batch or constant labels),
        # return a neutral score.
        return 0.5

    return float(np.mean(auc_scores))
