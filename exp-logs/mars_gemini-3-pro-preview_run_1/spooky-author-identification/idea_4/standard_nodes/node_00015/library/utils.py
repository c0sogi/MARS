import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for python, numpy, and torch to ensure reproducibility.

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


def clip_probabilities(probs):
    """
    Clips probabilities to the range [1e-15, 1 - 1e-15] as per the metric specification.

    Args:
        probs (np.ndarray): The predicted probabilities.

    Returns:
        np.ndarray: The clipped probabilities.
    """
    eps = 1e-15
    return np.clip(probs, eps, 1 - eps)


def compute_log_loss(y_true, y_pred):
    """
    Computes the multi-class logarithmic loss after normalizing and clipping probabilities.

    Args:
        y_true (np.ndarray or list): Ground truth labels (indices).
        y_pred (np.ndarray or list): Predicted probabilities (shape: [n_samples, n_classes]).

    Returns:
        float: The computed log loss.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 1. Rescale: each row is divided by the row sum
    row_sums = y_pred.sum(axis=1)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1.0
    y_pred = y_pred / row_sums[:, np.newaxis]

    # 2. Clip probabilities
    y_pred = clip_probabilities(y_pred)

    # 3. Compute Log Loss
    # We provide the list of labels to ensure log_loss knows there are 3 classes
    # even if a batch misses one.
    num_classes = y_pred.shape[1]
    labels = list(range(num_classes))

    return log_loss(y_true, y_pred, labels=labels)
