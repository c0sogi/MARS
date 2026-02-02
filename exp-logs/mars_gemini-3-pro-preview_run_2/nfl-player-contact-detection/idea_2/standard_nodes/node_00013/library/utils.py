import os
import random
import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef
from library.config import SEED


def seed_everything(seed: int = SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
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


def compute_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Matthews Correlation Coefficient.

    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_pred (np.ndarray): Predicted binary labels.

    Returns:
        float: The MCC score.
    """
    return matthews_corrcoef(y_true, y_pred)


def optimize_threshold(
    y_true: np.ndarray, y_pred_probs: np.ndarray, num_steps: int = 100
):
    """
    Finds the optimal probability threshold that maximizes MCC.

    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_pred_probs (np.ndarray): Predicted probabilities (0 to 1).
        num_steps (int): Number of steps in the grid search.

    Returns:
        tuple: (best_threshold, best_mcc_score)
    """
    thresholds = np.linspace(0.01, 0.99, num_steps)
    best_threshold = 0.5
    best_score = -1.0

    # Iterate through thresholds to find the best one
    for thresh in thresholds:
        # Convert probabilities to binary predictions based on current threshold
        y_pred_binary = (y_pred_probs >= thresh).astype(int)

        # Calculate MCC
        score = matthews_corrcoef(y_true, y_pred_binary)

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold, best_score
