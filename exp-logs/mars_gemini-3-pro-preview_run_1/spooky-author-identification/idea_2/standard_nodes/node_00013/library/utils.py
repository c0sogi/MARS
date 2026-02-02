import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_log_loss(y_true, y_pred, labels=None):
    """
    Computes the multi-class logarithmic loss.

    Args:
        y_true (array-like): Ground truth labels (indices or strings).
        y_pred (array-like): Predicted probabilities.
        labels (list, optional): List of class labels to index the probabilities.

    Returns:
        float: The calculated log loss.
    """
    # sklearn's log_loss handles label encoding if classes are provided
    return log_loss(y_true, y_pred, labels=labels)


def format_submission(ids, probs, columns):
    """
    Formats the predictions into a DataFrame for submission, applying
    the required probability clipping.

    Args:
        ids (list or array): Sequence of example IDs.
        probs (numpy.ndarray): Predicted probabilities of shape (n_samples, n_classes).
        columns (list): List of column names for the classes (e.g., ['EAP', 'HPL', 'MWS']).

    Returns:
        pd.DataFrame: The formatted submission DataFrame.
    """
    # Apply clipping to avoid extremes of the log function
    # Predicted probabilities are replaced with max(min(p, 1-10^-15), 10^-15)
    epsilon = 1e-15
    probs_clipped = np.clip(probs, epsilon, 1 - epsilon)

    # Create DataFrame
    submission = pd.DataFrame(probs_clipped, columns=columns)
    submission.insert(0, "id", ids)

    return submission
