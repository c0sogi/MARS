import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Ensures deterministic behavior for cuDNN backends.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_log_loss(y_true, y_pred, labels=None):
    """
    Calculates the Multi Class Log Loss using scikit-learn.

    Args:
        y_true (array-like): Ground truth (correct) labels for n_samples samples.
                             Can be class indices or string labels.
        y_pred (array-like): Predicted probabilities, as returned by a classifier's
                             predict_proba method. Shape (n_samples, n_classes).
        labels (array-like, optional): List of labels to index the classes in y_pred.
                                       This is required if y_true are string labels
                                       to ensure the columns of y_pred match the
                                       correct classes.

    Returns:
        float: The calculated log loss.
    """
    return log_loss(y_true, y_pred, labels=labels)
