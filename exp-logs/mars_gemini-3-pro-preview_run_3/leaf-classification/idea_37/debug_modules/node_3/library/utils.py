import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility
    across python, numpy, and torch.

    Args:
        seed (int): The seed value to use.
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


def calculate_log_loss(y_true, y_pred):
    """
    Calculates the multi-class log loss metric as per competition specifications.

    The probabilities are rescaled to sum to 1 per row, then clipped to the range
    [1e-15, 1 - 1e-15] before calculating the negative log likelihood.

    Args:
        y_true (array-like): Ground truth labels. Can be 1D array of integer labels
                             or 2D one-hot encoded array.
        y_pred (array-like): Predicted probabilities of shape (n_samples, n_classes).

    Returns:
        float: The calculated log loss.
    """
    # Convert to numpy array for operations
    y_pred = np.array(y_pred)

    # 1. Rescale rows to sum to 1
    # Handle cases where sum is 0 to avoid division by zero (though unlikely in valid predictions)
    row_sums = y_pred.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    y_pred = y_pred / row_sums[:, np.newaxis]

    # 2. Clip probabilities
    # Range: [10^-15, 1 - 10^-15]
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # 3. Calculate Log Loss
    # We define labels explicitly to handle batches that might not contain all classes
    # Assuming y_pred columns correspond to classes 0, 1, ..., K-1
    n_classes = y_pred.shape[1]
    labels = np.arange(n_classes)

    # Use sklearn's log_loss
    # eps is passed here as well, though we manually clipped above to be explicit
    loss = log_loss(y_true, y_pred, labels=labels)

    return loss
