import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

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

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_log_loss(y_true, y_pred):
    """
    Computes the Log Loss metric for the competition.

    Args:
        y_true (array-like): Ground truth (correct) labels. Can be soft probabilities
                             (n_samples, n_classes) or class indices.
        y_pred (array-like): Predicted probabilities, as returned by a classifier's
                             predict_proba method (n_samples, n_classes).

    Returns:
        float: The log loss.
    """
    # Using eps=1e-15 as standard for "auto" epsilon in log loss calculations
    return log_loss(y_true, y_pred, eps=1e-15)
