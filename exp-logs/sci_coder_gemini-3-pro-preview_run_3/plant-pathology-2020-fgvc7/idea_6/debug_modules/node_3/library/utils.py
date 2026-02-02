import os
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility by wrapping the Config utility.
    """
    Config.set_seed(seed)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking loss and metrics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_class_weights(load_cached_data=True, debug=Config.DEBUG):
    """
    Calculates inverse frequency class weights for the loss function based on the training set.
    Implements caching to avoid re-reading the CSV and re-computing on every run.

    Args:
        load_cached_data (bool): If True, attempts to load weights from the cache.
        debug (bool): Argument included for interface flexibility; currently weights are
                      always computed on the full training set for consistency.

    Returns:
        torch.Tensor: A tensor containing the weights for each class, moved to the configured device.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORK_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORK_DIR, "class_weights.npy")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            weights = np.load(cache_path)
            return torch.tensor(weights, dtype=torch.float32).to(Config.DEVICE)
        except Exception:
            # If loading fails, proceed to compute from scratch
            pass

    # 2. Compute from scratch
    if not os.path.exists(Config.TRAIN_CSV):
        # Fallback if file is missing (though it should exist)
        return torch.ones(len(Config.CLASSES)).to(Config.DEVICE)

    df = pd.read_csv(Config.TRAIN_CSV)

    counts = []
    for cls in Config.CLASSES:
        # Check if the class exists as a column (one-hot)
        if cls in df.columns:
            counts.append(df[cls].sum())
        # Fallback to checking stratify_label if one-hot columns are missing
        elif "stratify_label" in df.columns:
            counts.append((df["stratify_label"] == cls).sum())
        else:
            counts.append(0)

    counts = np.array(counts, dtype=np.float32)

    # Avoid division by zero
    counts = np.maximum(counts, 1.0)

    total_samples = np.sum(counts)
    num_classes = len(Config.CLASSES)

    # Calculate weights using the 'balanced' heuristic: n_samples / (n_classes * np.bincount(y))
    weights = total_samples / (num_classes * counts)

    # 3. Save to cache
    np.save(cache_path, weights)

    return torch.tensor(weights, dtype=torch.float32).to(Config.DEVICE)
