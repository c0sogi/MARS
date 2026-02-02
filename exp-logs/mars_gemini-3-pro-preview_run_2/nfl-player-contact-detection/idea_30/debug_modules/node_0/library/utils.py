import os
import random
import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the seed for reproducibility across Python, NumPy, and PyTorch.

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


def compute_mcc(y_true, y_pred):
    """
    Calculates the Matthews Correlation Coefficient between ground truth and predictions.

    Args:
        y_true (array-like): Ground truth binary labels (0 or 1).
        y_pred (array-like): Predicted binary labels (0 or 1).

    Returns:
        float: The MCC score.
    """
    # Ensure inputs are numpy arrays of integers
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    return matthews_corrcoef(y_true, y_pred)


def optimize_threshold(y_true, y_probs, step=0.01):
    """
    Performs a grid search to find the decision threshold that maximizes MCC.

    Args:
        y_true (array-like): Ground truth binary labels.
        y_probs (array-like): Predicted probabilities (values between 0 and 1).
        step (float): The step size for the grid search.

    Returns:
        tuple: (best_threshold, best_mcc)
            best_threshold (float): The threshold value that yielded the highest MCC.
            best_mcc (float): The maximum MCC score achieved.
    """
    best_mcc = -1.0
    best_threshold = 0.5

    # Generate thresholds from step to 1.0 (exclusive)
    thresholds = np.arange(step, 1.0, step)

    # Iterate through thresholds to find the optimum
    for thresh in thresholds:
        # Binarize probabilities based on current threshold
        y_pred = (y_probs >= thresh).astype(int)

        # Calculate MCC
        score = matthews_corrcoef(y_true, y_pred)

        if score > best_mcc:
            best_mcc = score
            best_threshold = thresh

    return best_threshold, best_mcc
