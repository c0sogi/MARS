import os
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    Delegates to the Config class implementation to avoid redundancy.

    Args:
        seed (int): The random seed value.
    """
    Config.seed_everything(seed)


def compute_class_weights(
    metadata_path, load_cached_data=True, debug=False, debug_subset_size=100
):
    """
    Computes class weights for handling dataset imbalance using the inverse frequency method.
    Implements caching to avoid re-computation on subsequent runs.

    Args:
        metadata_path (str): Path to the CSV file containing dataset metadata (must have a 'label' column).
        load_cached_data (bool): If True, attempts to load weights from the cache directory.
        debug (bool): If True, computes weights based on a subset of the data.
        debug_subset_size (int): Number of rows to use if debug is True.

    Returns:
        torch.Tensor: A tensor containing the weight for each class, moved to the configured device.
    """
    # Define the cache directory and file path
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Construct a cache filename. If debugging, do not use the main cache to avoid pollution.
    cache_filename = "class_weights_debug.npy" if debug else "class_weights.npy"
    cache_path = os.path.join(cache_dir, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            weights = np.load(cache_path)
            # Ensure weights are float32 for PyTorch loss functions
            return torch.from_numpy(weights).float().to(Config.DEVICE)
        except Exception:
            # If loading fails, proceed to computation
            pass

    # 2. Compute from scratch
    # Load metadata
    df = pd.read_csv(metadata_path)

    # Handle debug mode
    if debug:
        df = df.head(debug_subset_size)

    labels = df["label"].values

    # Determine number of classes and count frequencies
    # We assume classes are 0-indexed integers up to Config.NUM_CLASSES - 1
    num_classes = Config.NUM_CLASSES
    class_counts = np.bincount(labels, minlength=num_classes)

    # Total number of samples in the (subset of) dataset
    total_samples = len(labels)

    # Compute Inverse Frequency Weights
    # Formula: w_j = n_samples / (n_classes * n_samples_j)
    # Add a small epsilon to denominator to prevent division by zero
    weights = total_samples / (num_classes * (class_counts + 1e-6))

    # Cast to float32 numpy array
    weights = weights.astype(np.float32)

    # 3. Save to cache
    np.save(cache_path, weights)

    # Return as torch Tensor on the appropriate device
    return torch.from_numpy(weights).to(Config.DEVICE)
