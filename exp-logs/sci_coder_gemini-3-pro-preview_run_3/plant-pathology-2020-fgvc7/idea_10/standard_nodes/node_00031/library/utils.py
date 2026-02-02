import os
import numpy as np
import torch
from library.config import seed_everything as _lib_seed_everything
from library.config import get_class_weights as _lib_get_class_weights


def seed_everything(seed):
    """
    Sets the random seed for reproducibility across numpy, torch, and python random.
    Wraps the provided library function.

    Args:
        seed (int): The seed value to use.
    """
    _lib_seed_everything(seed)


def calculate_class_weights(df, target_cols, load_cached_data=True):
    """
    Calculates inverse frequency weights for class balancing in the loss function.
    Implements a caching mechanism to store/retrieve weights from disk.

    Args:
        df (pd.DataFrame): The training dataframe containing target labels.
        target_cols (list): List of column names representing the targets.
        load_cached_data (bool): If True, attempts to load weights from the cache.

    Returns:
        torch.Tensor: A tensor containing the weights for each class.
    """
    # Directory Safety
    cache_dir = "./working/idea_10/"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "class_weights.npy")

    # Logic Flow 1: Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            weights_np = np.load(cache_path)
            return torch.tensor(weights_np, dtype=torch.float32)
        except Exception:
            # If loading fails (corrupt file, etc.), proceed to calculation
            pass

    # Logic Flow 2: Compute/process the data from scratch
    # We use the logic already defined in library.config
    weights = _lib_get_class_weights(df, target_cols)

    # Save the result to the cache directory
    if isinstance(weights, torch.Tensor):
        weights_np = weights.cpu().numpy()
    else:
        weights_np = np.array(weights)

    np.save(cache_path, weights_np)

    # Ensure return type is torch.Tensor
    if not isinstance(weights, torch.Tensor):
        weights = torch.tensor(weights, dtype=torch.float32)

    return weights
