import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_class_weights(df, load_cached_data=True):
    """
    Calculates inverse frequency class weights to handle class imbalance.
    Implements caching mechanism using .npy format.

    Args:
        df (pd.DataFrame): Training dataframe containing labels.
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        torch.Tensor: Tensor of class weights on the configured device.
    """
    cache_path = Config.CLASS_WEIGHTS_PATH

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            weights = np.load(cache_path)
            return torch.FloatTensor(weights).to(Config.DEVICE)
        except Exception:
            # If loading fails, proceed to compute
            pass

    # 2. Compute weights
    # Ensure directory exists for saving later
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Determine class counts
    # We rely on Config.CLASS_LABELS to ensure correct order:
    # ["healthy", "multiple_diseases", "rust", "scab"]

    # If 'stratify_label' is available (from metadata), use it for counting
    if "stratify_label" in df.columns:
        counts_series = df["stratify_label"].value_counts()
    else:
        # Otherwise, derive active label from one-hot columns
        counts_series = df[Config.CLASS_LABELS].idxmax(axis=1).value_counts()

    total_samples = len(df)
    num_classes = len(Config.CLASS_LABELS)

    weights_list = []
    for label in Config.CLASS_LABELS:
        count = counts_series.get(label, 0)
        # Inverse frequency formula: N / (C * count)
        if count > 0:
            w = total_samples / (num_classes * count)
        else:
            # Fallback for missing classes in a subset (e.g. debug mode)
            w = 1.0
        weights_list.append(w)

    weights = np.array(weights_list, dtype=np.float32)

    # 3. Save to cache
    np.save(cache_path, weights)

    return torch.FloatTensor(weights).to(Config.DEVICE)


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the mean column-wise ROC AUC score.

    Args:
        y_true: Ground truth labels (N, NumClasses), numpy array or tensor.
        y_pred: Predicted probabilities (N, NumClasses), numpy array or tensor.

    Returns:
        float: The mean column-wise ROC AUC.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Calculate Macro ROC AUC
    # 'macro': Calculate metrics for each label, and find their unweighted mean.
    # This does not take label imbalance into account, which matches "Mean column-wise".
    try:
        score = roc_auc_score(y_true, y_pred, average="macro")
        # Cite debug_lesson_3: Explicitly Handle Silent NaNs
        if np.isnan(score):
            raise ValueError("Result is NaN")
    except ValueError:
        # Fallback: Calculate column-wise average ignoring invalid columns
        scores = []
        for i in range(y_true.shape[1]):
            try:
                if len(np.unique(y_true[:, i])) > 1:
                    scores.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
            except ValueError:
                continue

        if scores:
            score = np.mean(scores)
        else:
            score = 0.5

    return score
