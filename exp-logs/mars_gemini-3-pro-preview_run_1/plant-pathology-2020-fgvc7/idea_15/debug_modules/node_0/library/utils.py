import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import CFG


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


def calculate_class_weights(df, target_cols, load_cached_data=True):
    """
    Calculates class weights inversely proportional to class frequencies.
    Implements caching to ensure deterministic data processing and speed up subsequent runs.

    Args:
        df (pd.DataFrame): The training dataframe containing target columns.
        target_cols (list): List of target column names.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        torch.FloatTensor: A tensor of weights for each class on the configured device.
    """
    # Ensure working directory exists
    os.makedirs(CFG.working_dir, exist_ok=True)
    cache_path = os.path.join(CFG.working_dir, "class_weights.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            weights = np.load(cache_path)
            return torch.FloatTensor(weights).to(CFG.device)
        except Exception:
            # If load fails, proceed to compute
            pass

    # 2. Compute from scratch
    if df is None:
        raise ValueError("DataFrame is None and valid cache not found.")

    # Calculate counts for each class (sum of one-hot encoded columns)
    counts = df[target_cols].sum().values

    # Avoid division by zero (safety check, though unlikely in this dataset)
    counts = np.maximum(counts, 1)

    n_samples = len(df)
    n_classes = len(target_cols)

    # Formula: N / (K * n_j) for balanced weights
    weights = n_samples / (n_classes * counts)

    # 3. Save to cache
    np.save(cache_path, weights)

    return torch.FloatTensor(weights).to(CFG.device)


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the Mean Column-wise ROC AUC score.

    Args:
        y_true: True labels (N, C), can be numpy array or torch tensor.
        y_pred: Predicted probabilities (N, C), can be numpy array or torch tensor.

    Returns:
        float: The macro-averaged ROC AUC score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    try:
        # average='macro' computes AUC for each class and takes the unweighted mean
        score = roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        # Handle edge cases (e.g., only one class present in the batch/subset)
        score = 0.5

    return score


def save_submission(predictions, image_ids, target_cols, output_path):
    """
    Formats and saves the predictions to a CSV file.

    Args:
        predictions: (N, C) array or tensor of predicted probabilities.
        image_ids: List of image IDs corresponding to the predictions.
        target_cols: List of target column names.
        output_path: File path to save the submission CSV.
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()

    df_sub = pd.DataFrame(predictions, columns=target_cols)
    df_sub.insert(0, "image_id", image_ids)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sub.to_csv(output_path, index=False)
