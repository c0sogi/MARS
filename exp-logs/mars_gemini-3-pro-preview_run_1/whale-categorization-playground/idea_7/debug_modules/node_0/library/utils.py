import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducible results.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value of a metric.
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


def map5_metric(preds, targs):
    """
    Calculates the Mean Average Precision @ 5 (MAP@5) score.

    Args:
        preds (np.ndarray, torch.Tensor, or list): Shape (N, 5) containing predicted class indices or labels.
        targs (np.ndarray, torch.Tensor, or list): Shape (N,) containing true class indices or labels.

    Returns:
        float: The MAP@5 score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targs, torch.Tensor):
        targs = targs.detach().cpu().numpy()

    if len(preds) != len(targs):
        raise ValueError(
            f"Predictions ({len(preds)}) and targets ({len(targs)}) must have the same length."
        )

    score = 0.0
    for p, t in zip(preds, targs):
        # Convert p to list for easy indexing
        if isinstance(p, np.ndarray):
            p = p.tolist()

        # Handle scalar numpy/tensor items for t
        if hasattr(t, "item"):
            t = t.item()

        if t in p:
            # Rank is 0-indexed, score is 1/(rank+1)
            rank = p.index(t)
            if rank < 5:
                score += 1.0 / (rank + 1)

    return score / len(targs)


class WhaleLabelEncoder:
    """
    Encodes Whale string IDs to integers and decodes them back.
    Implements a caching mechanism using Parquet to ensure consistent mapping
    across different runs and between training/inference.
    """

    def __init__(self):
        self.classes_ = None
        self.class_to_idx = None

    def fit(self, ids, cache_path=None, load_cached_data=True):
        """
        Fits the encoder on a collection of IDs.

        Args:
            ids (list or pd.Series): List of whale IDs.
            cache_path (str, optional): Path to the cache file. Defaults to classes.parquet in working dir.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            self
        """
        if cache_path is None:
            cache_path = os.path.join(Config.WORKING_DIR, "classes.parquet")

        loaded = False

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                self.classes_ = df["Id"].values
                loaded = True
            except Exception:
                # If loading fails, proceed to compute
                pass

        # 2. Compute if not loaded
        if not loaded:
            # Get unique IDs and sort them for determinism
            unique_ids = sorted(list(set(ids)))
            self.classes_ = np.array(unique_ids)

            # Save to cache using Parquet (avoiding pickle)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df = pd.DataFrame({"Id": unique_ids})
            df.to_parquet(cache_path, index=False)

        # Build the lookup dictionary
        self.class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        return self

    def transform(self, ids):
        """
        Converts whale IDs to integer indices.
        """
        if self.class_to_idx is None:
            raise RuntimeError("LabelEncoder has not been fit yet.")

        # Handle single item
        if isinstance(ids, str):
            return self.class_to_idx[ids]

        return np.array([self.class_to_idx[x] for x in ids])

    def inverse_transform(self, indices):
        """
        Converts integer indices back to whale IDs.
        """
        if self.classes_ is None:
            raise RuntimeError("LabelEncoder has not been fit yet.")

        # Handle single item
        if np.isscalar(indices):
            return self.classes_[indices]

        return self.classes_[indices]

    def num_classes(self):
        """
        Returns the number of unique classes.
        """
        if self.classes_ is None:
            return 0
        return len(self.classes_)
