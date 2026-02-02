import os
import random
import numpy as np
import torch
from sklearn.preprocessing import LabelBinarizer
from library.config import RANDOM_SEED, CLIP_MIN, CLIP_MAX


def set_seed(seed=RANDOM_SEED):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use. Defaults to RANDOM_SEED from config.
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
    Calculates the Multi-class Log Loss with specific rescaling and clipping
    as defined in the task description.

    Steps:
    1. Rescale probabilities: each row is divided by the row sum.
    2. Clip probabilities: max(min(p, 1-10^-15), 10^-15).
    3. Compute Log Loss.

    Args:
        y_true (np.array): True labels. Can be 1D array of class indices or
                           2D one-hot encoded array.
        y_pred (np.array): Predicted probabilities (N_samples, N_classes).

    Returns:
        float: The calculated log loss.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 1. Rescale probabilities
    # "The submitted probabilities... are rescaled prior to being scored
    # (each row is divided by the row sum)"
    row_sums = y_pred.sum(axis=1)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1.0
    y_pred_rescaled = y_pred / row_sums[:, np.newaxis]

    # 2. Clip probabilities
    # "predicted probabilities are replaced with max(min(p,1-10^{-15}),10^{-15})"
    y_pred_clipped = np.clip(y_pred_rescaled, CLIP_MIN, CLIP_MAX)

    # 3. Compute Log Loss
    # Handle y_true being 1D (class indices) or 2D (one-hot)
    if y_true.ndim == 1:
        # If y_true are labels, we need to select the prob corresponding to the true class
        # We assume y_true contains integers 0 to N_classes-1

        # Check if y_true needs encoding (e.g. if string labels are passed, though pipeline should handle this)
        # For calculation, we gather the probabilities of the true classes

        # Create an index array for the rows
        rows = np.arange(len(y_true))

        # Extract the probabilities for the true classes
        true_class_probs = y_pred_clipped[rows, y_true]

        # Calculate log loss: -mean(log(p_true))
        score = -np.mean(np.log(true_class_probs))

    else:
        # If y_true is one-hot encoded (or soft labels)
        # Normalize y_true just in case, though usually one-hot sums to 1
        # Formula: - (1/N) * sum( sum( y_true_ij * log(y_pred_ij) ) )

        # We only sum over classes, then mean over samples
        score = -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))

    return score
