import os
import random
import numpy as np
import pickle
from sklearn.metrics import log_loss


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # Attempt to set torch seeds if the library is available
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def clip_probabilities(probas):
    """
    Clips probabilities to the range [1e-15, 1-1e-15] to ensure numerical stability
    and avoid undefined log(0) operations.

    Args:
        probas (np.ndarray): Array of predicted probabilities.

    Returns:
        np.ndarray: The clipped probabilities.
    """
    epsilon = 1e-15
    return np.clip(probas, epsilon, 1 - epsilon)


def calculate_log_loss(y_true, y_pred, labels=None):
    """
    Calculates the multi-class logarithmic loss.

    Args:
        y_true (array-like): True class labels or indices.
        y_pred (array-like): Predicted probabilities.
        labels (list, optional): List of class labels to index the probabilities.

    Returns:
        float: The calculated log loss.
    """
    # Clip probabilities before calculation to match submission constraints
    y_pred_clipped = clip_probabilities(y_pred)
    return log_loss(y_true, y_pred_clipped, labels=labels)


def save_pickle(obj, path):
    """
    Saves a Python object to a file using pickle.

    Args:
        obj: The object to save.
        path (str): The destination file path.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    """
    Loads a Python object from a pickle file.

    Args:
        path (str): The file path to load from.

    Returns:
        The loaded object.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "rb") as f:
        return pickle.load(f)
