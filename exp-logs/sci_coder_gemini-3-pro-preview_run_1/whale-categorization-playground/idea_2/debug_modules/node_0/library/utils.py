import os
import numpy as np
import pandas as pd
from library.config import Config, seed_everything

# Alias seed_everything to set_seed as required
set_seed = seed_everything


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


def apk(actual, predicted, k=5):
    """
    Computes the average precision at k.

    Args:
        actual: The ground truth value (scalar or list).
        predicted: List of predicted values.
        k: The number of top predictions to consider.

    Returns:
        The average precision at k.
    """
    # Ensure actual is a list for containment check
    if not isinstance(actual, (list, np.ndarray)):
        actual = [actual]

    # Truncate predictions to k
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=5):
    """
    Computes the mean average precision at k.

    Args:
        actual: List of ground truth values.
        predicted: List of lists of predicted values.
        k: The number of top predictions to consider.

    Returns:
        The mean average precision at k.
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def map5(targets, predictions):
    """
    Wrapper for Mean Average Precision @ 5.

    Args:
        targets: List of ground truth labels/indices.
        predictions: List of lists of predicted labels/indices (top 5).

    Returns:
        MAP@5 score.
    """
    return mapk(targets, predictions, k=5)


class IdEncoder:
    """
    Helper to map between string labels (Whale Ids) and integer indices.
    """

    def __init__(self, classes):
        self.classes_ = np.array(classes)
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

    def transform(self, labels):
        """
        Converts labels (strings) to indices (ints).
        Can handle single label or list/array of labels.
        """
        if isinstance(labels, (list, np.ndarray, pd.Series)):
            return np.array([self.class_to_idx.get(x, -1) for x in labels])
        return self.class_to_idx.get(labels, -1)

    def inverse_transform(self, indices):
        """
        Converts indices (ints) to labels (strings).
        Can handle single index or list/array of indices.
        """
        if isinstance(indices, (list, np.ndarray, pd.Series)):
            return self.classes_[indices]
        return self.classes_[indices]

    def num_classes(self):
        return len(self.classes_)


def get_id_encoder(load_cached_data=True):
    """
    Creates or loads an IdEncoder based on the training metadata.
    Implements caching to avoid re-reading the CSV every time.

    Args:
        load_cached_data (bool): If True, tries to load classes from cache.
                                 If False or cache missing, recomputes.

    Returns:
        IdEncoder: Fitted encoder instance.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "classes.npy")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    classes = None

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            # Load from npy
            classes = np.load(cache_path)
        except Exception:
            # Fallback if load fails
            classes = None

    # If classes not loaded, compute from scratch
    if classes is None:
        df = pd.read_csv(Config.TRAIN_CSV)
        unique_ids = df["Id"].unique()
        # Sort for deterministic mapping and cast to string
        classes = np.sort(unique_ids).astype(str)

        # Save to npy
        np.save(cache_path, classes)

    return IdEncoder(classes)
