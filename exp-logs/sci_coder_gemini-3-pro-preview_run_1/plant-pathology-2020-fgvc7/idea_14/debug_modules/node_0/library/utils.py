import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metric(y_true, y_pred):
    """
    Calculates the Mean Column-wise ROC AUC score.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth labels, shape (N, num_classes).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities, shape (N, num_classes).

    Returns:
        float: The mean column-wise ROC AUC score.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Handle NaNs or Infs in predictions to prevent crashes
    if np.isnan(y_pred).any() or np.isinf(y_pred).any():
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1.0, neginf=0.0)

    try:
        # Calculate Macro-Average ROC AUC
        # average='macro' computes the metric independently for each class and then takes the average
        score = roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        # Fallback for edge cases (e.g., a class has no positive samples in the current batch/fold)
        # We calculate the score for each valid column independently
        num_classes = y_true.shape[1]
        scores = []
        for i in range(num_classes):
            try:
                # Only calculate if the class has more than one unique value (both 0 and 1 present)
                if len(np.unique(y_true[:, i])) > 1:
                    s = roc_auc_score(y_true[:, i], y_pred[:, i])
                    scores.append(s)
            except ValueError:
                continue

        if scores:
            score = np.mean(scores)
        else:
            # If calculation fails entirely (e.g., all classes are constant), return 0.5
            score = 0.5

    return score
