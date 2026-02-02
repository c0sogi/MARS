import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

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


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the mean column-wise ROC AUC score.

    Args:
        y_true (np.ndarray or list): Ground truth labels.
                                     Can be shape (N,) [class indices] or (N, C) [one-hot/multilabel].
        y_pred (np.ndarray or list): Predicted probabilities of shape (N, C).

    Returns:
        float: The mean column-wise ROC AUC score.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # If y_true is 1D (class indices), convert to one-hot encoding
    if y_true.ndim == 1:
        n_values = Config.NUM_CLASSES
        # Initialize zero array
        y_true_one_hot = np.zeros((y_true.size, n_values))
        # Set appropriate indices to 1
        y_true_one_hot[np.arange(y_true.size), y_true.astype(int)] = 1
        y_true = y_true_one_hot

    # Calculate Macro ROC AUC
    # multi_class='ovr' (One-vs-Rest) is standard for multi-class ROC AUC
    try:
        score = roc_auc_score(y_true, y_pred, average="macro", multi_class="ovr")
    except ValueError:
        # This block handles cases where a class might be completely missing
        # from y_true in a small batch or specific fold split.
        # We calculate AUC for present classes and average them.
        scores = []
        for i in range(y_true.shape[1]):
            # Only calculate if both classes (0 and 1) are present
            if len(np.unique(y_true[:, i])) > 1:
                col_score = roc_auc_score(y_true[:, i], y_pred[:, i])
                scores.append(col_score)

        if scores:
            score = np.mean(scores)
        else:
            # Fallback if no classes can be evaluated (unlikely in proper validation)
            score = 0.5

    return score
