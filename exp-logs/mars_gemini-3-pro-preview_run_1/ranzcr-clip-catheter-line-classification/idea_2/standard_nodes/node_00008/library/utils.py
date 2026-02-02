import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed: int = Config.SEED) -> None:
    """
    Sets the random seed for python, numpy, and torch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Returns the PyTorch device to use for training/inference.

    Returns:
        torch.device: The computed device (CUDA or CPU).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def calculate_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the average Area Under the ROC Curve (AUC) for multi-label classification.
    Handles cases where specific columns might have only one class present (e.g., in debug mode)
    by skipping them in the average calculation to prevent errors.

    Args:
        y_true (np.ndarray): Ground truth binary labels of shape (N, num_classes).
        y_pred (np.ndarray): Predicted probabilities of shape (N, num_classes).

    Returns:
        float: The mean AUC score across all valid columns.
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    aucs = []
    num_classes = y_true.shape[1]

    for i in range(num_classes):
        # Extract column for current class
        y_true_col = y_true[:, i]
        y_pred_col = y_pred[:, i]

        # Check if the column has both classes (0 and 1)
        # roc_auc_score throws ValueError if only one class is present in y_true
        if len(np.unique(y_true_col)) == 2:
            try:
                score = roc_auc_score(y_true_col, y_pred_col)
                aucs.append(score)
            except ValueError:
                # Fallback if something unexpected happens
                pass
        else:
            # If only one class is present (e.g. all 0s), AUC is undefined.
            # We skip this column for the mean calculation to avoid skewing with arbitrary values.
            pass

    if len(aucs) == 0:
        return 0.5  # Default baseline if no valid columns found

    return np.mean(aucs)
