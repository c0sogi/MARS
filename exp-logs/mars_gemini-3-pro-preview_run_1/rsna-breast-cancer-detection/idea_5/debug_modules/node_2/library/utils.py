import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from library.config import TRAIN_METADATA_PATH, WORKING_DIR, seed_everything


def probabilistic_f1(y_true, y_pred, epsilon=1e-7):
    """
    Computes the Probabilistic F1 score (pF1) as defined in the task metric.

    pF1 = 2 * (pPrecision * pRecall) / (pPrecision + pRecall)
    pPrecision = pTP / (pTP + pFP)
    pRecall = pTP / (TP + FN)

    Args:
        y_true: Ground truth labels (binary). Can be a numpy array or torch Tensor.
        y_pred: Predicted probabilities (0 to 1). Can be a numpy array or torch Tensor.
        epsilon: Small constant to prevent division by zero.

    Returns:
        float: The probabilistic F1 score.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Flatten arrays to ensure 1D
    y_true = y_true.astype(float).flatten()
    y_pred = y_pred.astype(float).flatten()

    # Calculate Probabilistic True Positives (pTP)
    # pTP = sum(prediction * target)
    pTP = np.sum(y_pred * y_true)

    # Calculate Probabilistic False Positives (pFP)
    # pFP = sum(prediction * (1 - target))
    pFP = np.sum(y_pred * (1.0 - y_true))

    # Calculate Total Positives (TP + FN)
    # This is simply the sum of the binary targets
    total_positives = np.sum(y_true)

    # Calculate Probabilistic Precision
    # pPrecision = pTP / (pTP + pFP)
    # Note: pTP + pFP simplifies to sum(y_pred)
    sum_preds = np.sum(y_pred)
    pPrecision = pTP / (sum_preds + epsilon)

    # Calculate Probabilistic Recall
    # pRecall = pTP / (TP + FN)
    pRecall = pTP / (total_positives + epsilon)

    # Calculate Probabilistic F1
    pF1 = 2 * (pPrecision * pRecall) / (pPrecision + pRecall + epsilon)

    return float(pF1)


def get_age_scaler(load_cached_data=True):
    """
    Fits or loads a StandardScaler for the 'age' column.

    Implements a caching mechanism using .npy files to store mean and std,
    avoiding pickle as per requirements.

    Args:
        load_cached_data (bool): If True, attempts to load stats from cache.
                                 If False or cache miss, recomputes stats.

    Returns:
        sklearn.preprocessing.StandardScaler: A scaler with manually set mean and scale,
                                              ready to transform age data.
    """
    cache_path = os.path.join(WORKING_DIR, "age_stats.npy")

    mean = None
    std = None

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            stats = np.load(cache_path, allow_pickle=False)
            mean = stats[0]
            std = stats[1]
        except Exception:
            # If loading fails, proceed to recompute
            pass

    # 2. Compute if not loaded
    if mean is None or std is None:
        if not os.path.exists(TRAIN_METADATA_PATH):
            raise FileNotFoundError(
                f"Train metadata not found at {TRAIN_METADATA_PATH}"
            )

        df = pd.read_csv(TRAIN_METADATA_PATH)

        if "age" not in df.columns:
            raise ValueError("Column 'age' not found in training metadata.")

        # Extract age and compute stats, ignoring NaNs
        ages = df["age"].values
        mean = np.nanmean(ages)
        std = np.nanstd(ages)

        # Save to cache
        os.makedirs(WORKING_DIR, exist_ok=True)
        np.save(cache_path, np.array([mean, std]))

    # 3. Construct and return the scaler
    # We manually set attributes to avoid pickling the scaler object itself
    scaler = StandardScaler()
    scaler.mean_ = np.array([mean])
    scaler.scale_ = np.array([std])
    scaler.var_ = np.array([std**2])

    # Set auxiliary attributes required for the scaler to function
    scaler.n_samples_seen_ = np.array([1])  # Dummy value
    scaler.n_features_in_ = 1

    return scaler
