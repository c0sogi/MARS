import os
import random
import pickle
import joblib
import numpy as np
import torch
from scipy.stats import spearmanr


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_spearmanr(y_true, y_pred):
    """
    Computes the mean column-wise Spearman's correlation coefficient.

    Args:
        y_true (np.ndarray): Ground truth target values of shape (n_samples, n_targets).
        y_pred (np.ndarray): Predicted probabilities of shape (n_samples, n_targets).

    Returns:
        float: The mean Spearman's correlation coefficient across all targets.
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    n_targets = y_true.shape[1]
    corrs = []

    for i in range(n_targets):
        # Extract the i-th column
        t = y_true[:, i]
        p = y_pred[:, i]

        # Compute Spearman correlation
        # scipy.stats.spearmanr returns an object with 'statistic' attribute in newer versions
        # or a tuple (statistic, pvalue) in older versions.
        res = spearmanr(t, p)

        try:
            corr = res.statistic
        except AttributeError:
            corr = res[0]

        # Handle cases where correlation is undefined (e.g., constant input)
        if np.isnan(corr):
            corr = 0.0

        corrs.append(corr)

    return np.mean(corrs)


def save_pickle(obj, path):
    """
    Saves an object to a file using the pickle module.

    Args:
        obj: The object to save.
        path (str): The file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    """
    Loads an object from a pickle file.

    Args:
        path (str): The file path.

    Returns:
        The loaded object.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def save_joblib(obj, path):
    """
    Saves an object to a file using joblib.

    Args:
        obj: The object to save.
        path (str): The file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path):
    """
    Loads an object from a joblib file.

    Args:
        path (str): The file path.

    Returns:
        The loaded object.
    """
    return joblib.load(path)
