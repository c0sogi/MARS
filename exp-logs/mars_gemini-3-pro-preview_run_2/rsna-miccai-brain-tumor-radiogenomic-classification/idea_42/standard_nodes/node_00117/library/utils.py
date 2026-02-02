import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config, seed_everything


def set_seed(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Delegates to the implementation in library.config.

    Args:
        seed (int): The seed value to set. Defaults to Config.SEED.
    """
    seed_everything(seed)


def get_device() -> torch.device:
    """
    Returns the PyTorch device configured for the environment.

    Returns:
        torch.device: The device (CPU or CUDA) as defined in Config.DEVICE.
    """
    return torch.device(Config.DEVICE)


def compute_roc_auc(y_true, y_scores):
    """
    Computes the Area Under the ROC Curve (ROC AUC).

    Args:
        y_true (array-like): True binary labels.
        y_scores (array-like): Target scores or probability estimates of the positive class.

    Returns:
        float: The ROC AUC score. Returns 0.5 if only one class is present in y_true
               to prevent exceptions during batch-wise training/validation.
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Handle edge case where a batch might contain only one class
    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(y_true, y_scores)


def save_to_cache(data: np.ndarray, filename: str):
    """
    Saves a numpy array to the configured cache directory.

    Args:
        data (np.ndarray): The data array to save.
        filename (str): The filename (e.g., 'train_data.npy').
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    file_path = os.path.join(Config.CACHE_DIR, filename)
    np.save(file_path, data)


def load_from_cache(filename: str):
    """
    Attempts to load a numpy array from the configured cache directory.

    Args:
        filename (str): The filename to load (e.g., 'train_data.npy').

    Returns:
        np.ndarray or None: The loaded data if the file exists, otherwise None.
    """
    file_path = os.path.join(Config.CACHE_DIR, filename)
    if os.path.exists(file_path):
        # allow_pickle is False to ensure we only load numeric data for security and strictness
        return np.load(file_path, allow_pickle=False)
    return None


def print_metric(phase: str, metric_name: str, value: float):
    """
    Prints a metric value with full precision (no rounding).

    Args:
        phase (str): The phase of execution (e.g., 'Train', 'Val').
        metric_name (str): The name of the metric (e.g., 'AUC').
        value (float): The metric value.
    """
    print(f"{phase} {metric_name}: {value}")
