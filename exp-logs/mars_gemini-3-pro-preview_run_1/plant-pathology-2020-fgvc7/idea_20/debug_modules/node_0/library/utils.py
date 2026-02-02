import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_class_weights(df, load_cached_data=True):
    """
    Calculates inverse class frequency weights for the loss function.

    Args:
        df (pd.DataFrame): The training dataframe containing target columns.
        load_cached_data (bool): Whether to try loading cached weights from disk.

    Returns:
        torch.Tensor: A tensor of shape (num_classes,) containing the weights.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "class_weights.npy")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            weights_np = np.load(cache_path)
            return torch.tensor(weights_np, dtype=torch.float32).to(Config.DEVICE)
        except Exception:
            # If loading fails, proceed to recalculate
            pass

    # 2. Compute from scratch
    # Identify the target class for each row.
    # We assume targets are one-hot encoded or probabilities in the columns defined in Config.CLASSES.
    # We use argmax to get the dominant class index.
    target_cols = Config.CLASSES

    # Ensure columns exist
    missing_cols = [c for c in target_cols if c not in df.columns]
    if missing_cols:
        # Fallback: check if 'stratify_label' exists (from metadata generation)
        if "stratify_label" in df.columns:
            # Map string labels to indices
            label_to_idx = {label: idx for idx, label in enumerate(target_cols)}
            y_indices = df["stratify_label"].map(label_to_idx).values
        else:
            raise ValueError(
                f"Missing target columns {missing_cols} and 'stratify_label' in dataframe."
            )
    else:
        # Use argmax on the target columns
        y_indices = np.argmax(df[target_cols].values, axis=1)

    # Calculate counts
    class_counts = np.bincount(y_indices, minlength=len(target_cols))
    total_samples = len(y_indices)
    num_classes = len(target_cols)

    # Calculate weights: N / (C * n_j)
    # Add a small epsilon to avoid division by zero if a class is missing (unlikely in this dataset)
    weights = total_samples / (num_classes * (class_counts + 1e-6))

    # Normalize weights so they sum to num_classes (optional, but keeps scale consistent)
    # Or keep as is. Standard sklearn heuristic is n_samples / (n_classes * np.bincount(y))

    weights_np = np.array(weights, dtype=np.float32)

    # 3. Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, weights_np)

    return torch.tensor(weights_np, dtype=torch.float32).to(Config.DEVICE)
