import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_binary_targets(df):
    """
    Converts the original 4-class metadata into the 2 binary targets
    used for the Multi-Label Decomposition strategy.

    Target 1 (Rust): Present if 'rust' OR 'multiple_diseases' is True.
    Target 2 (Scab): Present if 'scab' OR 'multiple_diseases' is True.
    """
    # Ensure we are working with numeric types
    rust = df["rust"].values.astype(float)
    scab = df["scab"].values.astype(float)
    multiple = df["multiple_diseases"].values.astype(float)

    # Logical OR to create binary targets
    # Using np.maximum is equivalent to logical OR for 0/1 encoding
    binary_rust = np.maximum(rust, multiple)
    binary_scab = np.maximum(scab, multiple)

    return np.stack([binary_rust, binary_scab], axis=1)


def get_class_weights(df, load_cached_data=True):
    """
    Calculates positive weights for BCEWithLogitsLoss based on class imbalance.
    Implements caching mechanism as required.

    Args:
        df (pd.DataFrame): Training metadata.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        torch.Tensor: Weights for [rust, scab] classes.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "class_weights.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            weights = np.load(cache_path)
            return torch.tensor(weights, dtype=torch.float32)
        except Exception:
            pass  # Fallback to computation if load fails

    # 2. Compute from scratch
    # Get the binary targets used for training
    targets = get_binary_targets(df)  # Shape (N, 2)

    # Calculate weights: num_neg / num_pos
    pos_counts = np.sum(targets, axis=0)
    total_counts = len(targets)
    neg_counts = total_counts - pos_counts

    # Avoid division by zero
    pos_counts = np.maximum(pos_counts, 1)

    weights = neg_counts / pos_counts

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    np.save(cache_path, weights)

    return torch.tensor(weights, dtype=torch.float32)


def reconstruct_4_class_probabilities(rust_prob, scab_prob):
    """
    Reconstructs the 4-class probabilities from the 2 independent binary probabilities
    (Rust and Scab) based on the decomposition logic.

    Args:
        rust_prob (np.array): Probability of Rust (N,)
        scab_prob (np.array): Probability of Scab (N,)

    Returns:
        np.array: (N, 4) array corresponding to [healthy, multiple_diseases, rust, scab]
        Note: The order must match Config.ORIGINAL_TARGET_COLS
    """
    # P(Healthy) = (1 - Pr)(1 - Ps)
    healthy = (1 - rust_prob) * (1 - scab_prob)

    # P(Multiple) = Pr * Ps
    multiple = rust_prob * scab_prob

    # P(Rust_Only) = Pr * (1 - Ps)
    rust_only = rust_prob * (1 - scab_prob)

    # P(Scab_Only) = (1 - Pr) * Ps
    scab_only = (1 - rust_prob) * scab_prob

    # Stack in the order: healthy, multiple_diseases, rust, scab
    # This matches the sample_submission.csv column order usually,
    # but let's check Config.ORIGINAL_TARGET_COLS to be safe.
    # Config.ORIGINAL_TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

    return np.stack([healthy, multiple, rust_only, scab_only], axis=1)


def calculate_metric(y_true, y_pred):
    """
    Computes the Mean Column-wise ROC AUC.

    Args:
        y_true (np.array): Ground truth labels (N, 4) or (N, num_classes).
        y_pred (np.array): Predicted probabilities (N, 4) or (N, num_classes).

    Returns:
        float: The mean ROC AUC score.
    """
    try:
        # average='macro' calculates metrics for each label, and finds their unweighted mean.
        score = roc_auc_score(y_true, y_pred, average="macro")
        return score
    except ValueError:
        # Handle cases where a class might not be present in the batch/subset
        return 0.5


def save_checkpoint(state, filename="checkpoint.pth"):
    """
    Saves the model state to the checkpoint directory.
    """
    filepath = os.path.join(Config.CHECKPOINT_DIR, filename)
    torch.save(state, filepath)


def load_checkpoint(filename, model, optimizer=None, scheduler=None, device="cpu"):
    """
    Loads model state from a checkpoint.
    """
    filepath = os.path.join(Config.CHECKPOINT_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found at {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint.get("best_score", 0.0), checkpoint.get("epoch", 0)


def get_device():
    return torch.device(Config.DEVICE)
