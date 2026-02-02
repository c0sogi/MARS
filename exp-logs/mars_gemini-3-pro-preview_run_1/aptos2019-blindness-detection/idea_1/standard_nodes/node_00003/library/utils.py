import os
import random
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ordinal_encode(target, num_classes=Config.NUM_CLASSES):
    """
    Encodes an integer target into an ordinal binary vector.

    For N classes, we generate N-1 binary units.
    Unit k represents P(y > k).

    Example (5 classes):
    Target 0 -> [0, 0, 0, 0]
    Target 1 -> [1, 0, 0, 0]
    Target 2 -> [1, 1, 0, 0]
    Target 4 -> [1, 1, 1, 1]

    Args:
        target (int): The class label (0 to num_classes-1).
        num_classes (int): Total number of classes. Defaults to Config.NUM_CLASSES.

    Returns:
        torch.Tensor: A tensor of shape (num_classes - 1,) containing 0s and 1s.
    """
    # Number of ordinal units is num_classes - 1
    num_units = num_classes - 1
    target_vec = torch.zeros(num_units, dtype=torch.float32)

    # If target is 0, vector is all zeros.
    # If target is k, the first k units are 1.
    if target > 0:
        # Clamp target to ensure we don't index out of bounds
        k = min(target, num_units)
        target_vec[:k] = 1.0

    return target_vec


def ordinal_decode(probs):
    """
    Decodes ordinal probabilities into integer class labels.

    The final predicted score is obtained by summing the probabilities of the
    ordinal units and rounding to the nearest integer.

    Args:
        probs (torch.Tensor or np.ndarray): Probabilities for the ordinal units (after Sigmoid).
                                            Shape (N, num_units) or (num_units,).

    Returns:
        np.ndarray or int: The predicted class labels.
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()

    # Sum probabilities across the units axis
    if probs.ndim == 1:
        pred_sum = np.sum(probs)
        return int(np.round(pred_sum))
    else:
        # Sum across the last axis (units)
        pred_sum = np.sum(probs, axis=1)
        return np.round(pred_sum).astype(int)


def compute_qwk(y_true, y_pred):
    """
    Calculates the Quadratic Weighted Kappa score.

    Args:
        y_true (np.ndarray or list): Ground truth labels.
        y_pred (np.ndarray or list): Predicted labels.

    Returns:
        float: The QWK score.
    """
    # Ensure inputs are numpy arrays of integers
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    return cohen_kappa_score(y_true, y_pred, weights="quadratic")
