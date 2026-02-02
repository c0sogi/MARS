import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss


def seed_everything(seed):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to the value in config if not provided,
                    but here we accept it as an argument for flexibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU, though we have one.

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metric(y_true, y_pred, labels=None):
    """
    Calculates the Multi-Class Log Loss using scikit-learn's implementation.

    Args:
        y_true (array-like): Ground truth (correct) labels. Can be class indices or names.
        y_pred (array-like): Predicted probabilities, returned by the classifier.
        labels (array-like, optional): If provided, used to associate columns of y_pred
                                       with classes.

    Returns:
        float: The calculated log loss.
    """
    # Scikit-learn's log_loss handles multiclass classification.
    # We return the raw float value for full precision.
    return log_loss(y_true, y_pred, labels=labels)
