import os
import random
import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across various libraries.

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


def compute_mcc(y_true, y_pred):
    """
    Computes the Matthews Correlation Coefficient (MCC).

    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred (array-like): Predicted binary labels.

    Returns:
        float: The MCC score.
    """
    return matthews_corrcoef(y_true, y_pred)


def optimize_threshold(y_true, y_probs, steps=100):
    """
    Performs a grid search to find the probability threshold that maximizes MCC.

    Args:
        y_true (array-like): Ground truth binary labels.
        y_probs (array-like): Predicted probabilities (between 0 and 1).
        steps (int): Number of threshold steps to evaluate. Defaults to 100.

    Returns:
        tuple: (best_threshold, best_score) where best_threshold is the optimal
               cutoff and best_score is the corresponding MCC.
    """
    # Generate thresholds from 0.01 to 0.99
    thresholds = np.linspace(0.01, 0.99, steps)

    best_score = -1.0
    best_threshold = 0.5

    # Ensure inputs are numpy arrays for efficient processing
    y_true_np = np.array(y_true)
    y_probs_np = np.array(y_probs)

    for thresh in thresholds:
        # Binarize predictions based on current threshold
        y_pred = (y_probs_np >= thresh).astype(int)

        # Calculate metric
        score = compute_mcc(y_true_np, y_pred)

        if score > best_score:
            best_score = score
            best_threshold = thresh

    # Print full precision metrics as required
    print(f"Best Threshold: {best_threshold}")
    print(f"Best MCC: {best_score}")

    return best_threshold, best_score
