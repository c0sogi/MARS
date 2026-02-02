import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_class_weights(metadata_path, target_columns, load_cached_data=True):
    """
    Calculates class weights inversely proportional to class frequencies.
    Implements caching to avoid re-computation.

    Args:
        metadata_path (str): Path to the training metadata CSV.
        target_columns (list): List of target column names corresponding to classes.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        torch.Tensor: A tensor of class weights on the configured device.
    """
    # Define cache path in the working directory
    cache_path = os.path.join(Config.WORKING_DIR, "class_weights.npy")

    # Ensure the directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            weights = np.load(cache_path)
            return torch.tensor(weights, dtype=torch.float32).to(Config.DEVICE)
        except Exception:
            # If loading fails, proceed to compute
            pass

    # 2. Compute from scratch
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    df = pd.read_csv(metadata_path)

    # Calculate counts for each class
    # Assumes target columns are binary (One-Hot) or probability distributions
    class_counts = []
    for col in target_columns:
        if col in df.columns:
            class_counts.append(df[col].sum())
        else:
            # Handle missing columns gracefully, though unexpected
            class_counts.append(0)

    class_counts = np.array(class_counts)

    # Calculate weights: n_samples / (n_classes * n_samples_j)
    # This is the standard sklearn 'balanced' heuristic
    total_samples = np.sum(class_counts)
    n_classes = len(target_columns)

    # Add epsilon to avoid division by zero
    weights = total_samples / (n_classes * class_counts + 1e-6)

    # 3. Save to cache
    np.save(cache_path, weights)

    return torch.tensor(weights, dtype=torch.float32).to(Config.DEVICE)
