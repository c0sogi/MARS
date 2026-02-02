import os
import random
import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef


def seed_everything(seed: int):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def optimize_threshold(y_true, y_pred_probs):
    """
    Performs a grid search on validation probabilities to find the decision threshold
    that maximizes the Matthews Correlation Coefficient (MCC).

    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred_probs (array-like): Predicted probabilities (between 0 and 1).

    Returns:
        tuple: (best_threshold, best_mcc)
    """
    # Convert to numpy arrays for efficiency
    y_true = np.asarray(y_true)
    y_pred_probs = np.asarray(y_pred_probs)

    # Define the search space for thresholds
    thresholds = np.arange(0.01, 1.00, 0.01)

    best_mcc = -1.0
    best_threshold = 0.5

    for thresh in thresholds:
        # Binarize predictions based on current threshold
        y_pred = (y_pred_probs >= thresh).astype(int)

        # Calculate MCC
        # Note: If the classifier predicts only one class, MCC is 0.0
        if len(np.unique(y_pred)) < 2:
            mcc = 0.0
        else:
            mcc = matthews_corrcoef(y_true, y_pred)

        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = thresh

    print("Optimization Results:")
    print("Best Threshold:", best_threshold)
    print("Best MCC:", best_mcc)

    return best_threshold, best_mcc
