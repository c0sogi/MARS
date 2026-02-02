import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Enforces deterministic behavior in CuDNN.

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


def calculate_metric(y_true, y_pred):
    """
    Calculates the mean column-wise ROC AUC score.

    Args:
        y_true (np.ndarray): Ground truth labels (one-hot or binary indicators), shape (N, num_classes).
        y_pred (np.ndarray): Predicted probabilities, shape (N, num_classes).

    Returns:
        float: The mean column-wise ROC AUC.
    """
    # Use macro average to compute the metric for each label and find their unweighted mean
    try:
        score = roc_auc_score(y_true, y_pred, average="macro")
        if np.isnan(score):
            score = 0.0
    except ValueError:
        # Handle edge cases where a specific class might be missing in a small batch
        score = 0.0
    return score


def get_class_weights(load_cached_data=True):
    """
    Calculates inverse frequency class weights from the training dataframe.
    Implements a caching mechanism using .npy files.

    Args:
        load_cached_data (bool): If True, attempts to load weights from the cache.

    Returns:
        np.ndarray: Array of weights corresponding to Config.CLASS_LABELS.
    """
    cache_path = os.path.join(Config.WORK_DIR, "class_weights.npy")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            weights = np.load(cache_path)
            return weights
        except Exception:
            # If loading fails, proceed to compute
            pass

    # 2. Compute from scratch
    # Load training metadata
    df = pd.read_csv(Config.TRAIN_CSV)

    class_labels = Config.CLASS_LABELS
    counts = []

    # Calculate sample count for each class
    for label in class_labels:
        if label in df.columns:
            counts.append(df[label].sum())
        else:
            # Should not happen given metadata guarantees, but safe fallback
            counts.append(0)

    counts = np.array(counts)
    total_samples = len(df)
    num_classes = len(class_labels)

    # Calculate Inverse Frequency Weights: W_j = N / (C * N_j)
    # Add epsilon to prevent division by zero
    weights = total_samples / (num_classes * (counts + 1e-6))

    # 3. Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, weights)

    return weights
