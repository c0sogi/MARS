import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed=42):
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_class_weights(df, target_cols, load_cached_data=True):
    """
    Calculates class weights inversely proportional to class frequencies.
    Implements caching to store/retrieve the weights from disk.

    Args:
        df (pd.DataFrame): The training dataframe containing target columns.
        target_cols (list): List of column names corresponding to the targets.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.ndarray: Array of weights corresponding to the order of target_cols.
    """
    # Define cache path
    cache_dir = Config.WORKING_DIR
    cache_path = os.path.join(cache_dir, "class_weights.npy")

    # Ensure directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            weights = np.load(cache_path)
            # Verify shape matches current target_cols length to avoid stale cache issues
            if len(weights) == len(target_cols):
                return weights
        except Exception:
            # If loading fails, proceed to calculation
            pass

    # 2. Compute from scratch
    # We assume the columns in df are binary/one-hot or counts.
    # Calculate total number of samples
    total_samples = len(df)

    # Calculate count for each class
    class_counts = []
    for col in target_cols:
        # Sum the column to get the number of positive instances for this class
        count = df[col].sum()
        class_counts.append(count)

    class_counts = np.array(class_counts)

    # Avoid division by zero
    class_counts = np.where(class_counts == 0, 1, class_counts)

    # Calculate inverse frequency weights: Total / Count
    # This balances the contribution of each class to the loss
    weights = total_samples / class_counts

    # Normalize weights so they sum to the number of classes (optional but keeps loss scale similar)
    # weights = weights / weights.sum() * len(target_cols)

    # Cast to float32 for PyTorch compatibility
    weights = weights.astype(np.float32)

    # 3. Save to cache
    try:
        np.save(cache_path, weights)
    except Exception:
        pass  # Non-critical failure

    return weights
