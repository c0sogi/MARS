import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the Mean Column-wise ROC AUC score.

    Args:
        y_true: Ground truth labels (N, C), can be numpy array or torch.Tensor.
        y_pred: Predicted probabilities (N, C), can be numpy array or torch.Tensor.

    Returns:
        float: The macro-averaged ROC AUC score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    try:
        # Calculate Macro-Averaged ROC AUC
        score = roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        # Handle edge cases (e.g., only one class present in y_true during debugging/small batches)
        score = 0.0

    return score


def calculate_class_weights(metadata_path, target_cols=None, load_cached_data=True):
    """
    Computes inverse class frequency weights from the training metadata.
    Implements caching to ./working/idea_7/class_weights.npy.

    Args:
        metadata_path (str): Path to the training metadata CSV.
        target_cols (list, optional): List of target column names. Defaults to Config.TARGET_COLS.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        torch.Tensor: A tensor of shape (C,) containing the class weights, on Config.DEVICE.
    """
    if target_cols is None:
        target_cols = Config.TARGET_COLS

    # Ensure working directory exists
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "class_weights.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            weights_np = np.load(cache_path)
            return torch.tensor(weights_np, dtype=torch.float32).to(Config.DEVICE)
        except Exception:
            # If loading fails, proceed to recompute
            pass

    # 2. Compute from scratch
    df = pd.read_csv(metadata_path)

    # Calculate counts for each class
    # Assuming metadata has one-hot encoded columns or we can infer from them
    class_counts = df[target_cols].sum().values

    total_samples = len(df)
    num_classes = len(target_cols)

    # Inverse frequency weights: N / (C * N_c)
    # Add epsilon to avoid division by zero
    weights_np = total_samples / (num_classes * (class_counts + 1e-6))

    # 3. Save to cache
    np.save(cache_path, weights_np)

    return torch.tensor(weights_np, dtype=torch.float32).to(Config.DEVICE)
