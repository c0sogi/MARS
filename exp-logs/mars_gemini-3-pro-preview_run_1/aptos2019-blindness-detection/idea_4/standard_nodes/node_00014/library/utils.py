import os
import random
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def quadratic_weighted_kappa(y_true, y_pred):
    """
    Calculates the Quadratic Weighted Kappa metric.

    Args:
        y_true: Array-like of ground truth labels (integers 0-4).
        y_pred: Array-like of predicted labels (integers 0-4).

    Returns:
        float: The QWK score.
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Calculate score using sklearn
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def decode_ordinal_predictions(predictions):
    """
    Decodes ordinal regression predictions (probabilities) into discrete labels.

    The strategy is to sum the sigmoid probabilities for the ordinal classes
    and round to the nearest integer.

    Args:
        predictions: Tensor or array of shape (Batch_Size, 4) containing
                     sigmoid probabilities.

    Returns:
        numpy.ndarray: Array of shape (Batch_Size,) containing integer labels 0-4.
    """
    # Convert to numpy if tensor
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()

    # Sum probabilities across the class dimension (axis 1)
    # This assumes the input is [P(y>0), P(y>1), P(y>2), P(y>3)]
    # The sum gives the expected rank/score.
    continuous_scores = np.sum(predictions, axis=1)

    # Round to nearest integer
    discrete_labels = np.round(continuous_scores).astype(int)

    # Clip to ensure valid range [0, 4] (though sum of 4 sigmoids is naturally in [0, 4])
    discrete_labels = np.clip(discrete_labels, 0, 4)

    return discrete_labels
