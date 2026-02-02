import os
import random
import numpy as np
import torch
import logging
from sklearn.metrics import roc_auc_score


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name="cactus_classifier"):
    """
    Returns a configured logger instance.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    # Clear existing handlers to prevent duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve.

    Args:
        y_true: Ground truth labels (binary).
        y_pred: Predicted probabilities for the positive class.

    Returns:
        float: ROC AUC score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Handle potential edge case where batch has only one class
    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(y_true, y_pred)


class FileSizeScaler:
    """
    Utility class to normalize and denormalize file sizes for the regression task.
    Uses log1p transformation followed by min-max scaling.
    """

    # Constants derived from dataset analysis to cover range of ~200B to ~5000B
    MIN_LOG_VAL = 5.0
    MAX_LOG_VAL = 9.0

    @staticmethod
    def transform(file_sizes):
        """
        Transforms raw file sizes (bytes) to normalized regression targets [0, 1].

        Args:
            file_sizes: Array-like of file sizes in bytes.

        Returns:
            np.array: Normalized values.
        """
        file_sizes = np.array(file_sizes, dtype=np.float32)
        log_sizes = np.log1p(file_sizes)
        norm_sizes = (log_sizes - FileSizeScaler.MIN_LOG_VAL) / (
            FileSizeScaler.MAX_LOG_VAL - FileSizeScaler.MIN_LOG_VAL
        )
        return np.clip(norm_sizes, 0.0, 1.0)

    @staticmethod
    def inverse_transform(norm_sizes):
        """
        Transforms normalized regression targets back to file sizes (bytes).

        Args:
            norm_sizes: Array-like of normalized values.

        Returns:
            np.array: File sizes in bytes.
        """
        norm_sizes = np.array(norm_sizes, dtype=np.float32)
        log_sizes = (
            norm_sizes * (FileSizeScaler.MAX_LOG_VAL - FileSizeScaler.MIN_LOG_VAL)
            + FileSizeScaler.MIN_LOG_VAL
        )
        return np.expm1(log_sizes)


def get_file_sizes_with_cache(file_paths, base_dir, cache_path, load_cached_data=True):
    """
    Extracts file sizes for a list of file paths, with caching mechanism.

    Args:
        file_paths (list): List of relative file paths.
        base_dir (str): Base directory containing the files.
        cache_path (str): Path to save/load the .npy cache file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.array: Array of file sizes in bytes.
    """
    # Ensure cache directory exists
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception:
            pass  # Fallback to computation if load fails

    # 2. Compute from scratch
    sizes = []
    for rel_path in file_paths:
        full_path = os.path.join(base_dir, rel_path)
        if os.path.exists(full_path):
            sizes.append(os.path.getsize(full_path))
        else:
            sizes.append(0)  # Should not happen given metadata validation

    sizes_array = np.array(sizes, dtype=np.float32)

    # 3. Save to cache
    try:
        np.save(cache_path, sizes_array)
    except Exception:
        pass  # Non-critical failure

    return sizes_array
