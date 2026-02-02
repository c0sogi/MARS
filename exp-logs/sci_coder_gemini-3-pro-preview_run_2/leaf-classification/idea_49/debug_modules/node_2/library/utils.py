import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
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


def clipped_log_loss(y_true, y_pred):
    """
    Calculates the multi-class log loss with specific clipping and normalization
    as defined in the competition metric.

    The metric requires:
    1. Rescaling predicted probabilities so each row sums to 1.
    2. Clipping probabilities to the range [1e-15, 1-1e-15].
    3. Computing the negative log likelihood.

    Args:
        y_true: Ground truth labels. Can be a 1D array of labels/indices
                or a 2D one-hot encoded array.
        y_pred: Predicted probabilities. A 2D array of shape (n_samples, n_classes).

    Returns:
        float: The calculated log loss.
    """
    # Ensure predictions are float64 for precision at the metric floor
    y_pred = np.array(y_pred, dtype=np.float64)

    # Rescale prior to scoring: each row is divided by the row sum
    # We handle potential zero sums to avoid division by zero/NaNs,
    # though valid probability outputs should not sum to 0.
    row_sums = y_pred.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    y_pred = y_pred / row_sums[:, np.newaxis]

    # Clip probabilities to [1e-15, 1-1e-15] as specified in the metric description
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # Calculate log loss
    # We rely on sklearn's implementation for the mathematical calculation
    # sklearn's log_loss also has an 'eps' parameter, but we perform the
    # specific clipping and normalization explicitly above to ensure
    # exact compliance with the task description.
    score = log_loss(y_true, y_pred)

    return score
