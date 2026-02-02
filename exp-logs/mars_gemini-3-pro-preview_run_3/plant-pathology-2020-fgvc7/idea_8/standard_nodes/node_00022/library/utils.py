import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility by delegating to Config.seed_everything.
    """
    Config.seed_everything(seed)


def get_class_weights(load_cached_data: bool = True):
    """
    Calculates inverse frequency class weights for the loss function.
    Implements caching to avoid re-computing from CSV on every run.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        torch.FloatTensor: Tensor containing weights for each class on the configured device.
    """
    cache_path = os.path.join(Config.OUTPUT_DIR, "class_weights.npy")

    # Ensure output directory exists
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            weights = np.load(cache_path)
            return torch.FloatTensor(weights).to(Config.DEVICE)
        except Exception:
            # If load fails, proceed to compute
            pass

    # 2. Compute data from scratch
    if not os.path.exists(Config.TRAIN_CSV):
        raise FileNotFoundError(f"Train metadata not found at {Config.TRAIN_CSV}")

    train_df = pd.read_csv(Config.TRAIN_CSV)

    # Calculate counts for each class in the order defined in Config.CLASSES
    # Metadata contains one-hot encoded columns matching class names
    class_counts = []
    for cls in Config.CLASSES:
        if cls not in train_df.columns:
            raise ValueError(f"Class column '{cls}' not found in metadata.")
        class_counts.append(train_df[cls].sum())

    class_counts = np.array(class_counts)
    n_samples = len(train_df)
    n_classes = len(Config.CLASSES)

    # Calculate inverse frequency weights: n_samples / (n_classes * class_count)
    # Adding a small epsilon to avoid division by zero
    weights = n_samples / (n_classes * (class_counts + 1e-6))

    # 3. Save to cache
    np.save(cache_path, weights)

    return torch.FloatTensor(weights).to(Config.DEVICE)


def calculate_metric(y_true, y_pred):
    """
    Computes the Mean Column-wise ROC AUC score.

    Args:
        y_true: Ground truth labels (N, num_classes), can be numpy array or torch tensor.
        y_pred: Predicted probabilities (N, num_classes), can be numpy array or torch tensor.

    Returns:
        float: The mean ROC AUC score.
    """
    # Convert tensors to numpy arrays if necessary
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()

    # Calculate Macro-Average ROC AUC
    # 'macro': Calculate metrics for each label, and find their unweighted mean.
    try:
        score = roc_auc_score(y_true, y_pred, average="macro")
        if np.isnan(score):
            raise ValueError("ROC AUC score is NaN")
    except ValueError:
        # Handle edge cases (e.g., batch contains only one class), return neutral score
        score = 0.5

    return score
