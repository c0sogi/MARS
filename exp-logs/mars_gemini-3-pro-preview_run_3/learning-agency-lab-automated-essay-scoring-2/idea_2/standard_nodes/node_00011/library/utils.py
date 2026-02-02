import os
import random
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from library.config import Config


def seed_everything(seed: int = Config.SEED):
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

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_qwk(y_true, y_pred):
    """
    Computes the Quadratic Weighted Kappa (QWK) score.

    Handles both integer and continuous predictions by rounding and clipping
    continuous values to the valid score range [1, 6].

    Args:
        y_true (array-like): True scores.
        y_pred (array-like): Predicted scores (can be floats).

    Returns:
        float: The QWK score.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Handle continuous predictions from regression models
    # Round to nearest integer
    if np.issubdtype(y_pred.dtype, np.floating):
        y_pred = np.round(y_pred)

    # Clip predictions to the valid range [1, 6] and convert to integers
    y_pred = np.clip(y_pred, 1, 6).astype(int)
    y_true = y_true.astype(int)

    return cohen_kappa_score(y_true, y_pred, weights="quadratic")
