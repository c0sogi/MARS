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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_class_weights(
    df, target_cols, load_cached_data=True, cache_dir="./working/idea_12"
):
    """
    Calculates class weights using inverse frequency to handle class imbalance.
    Implements caching mechanism using .npy format.

    Args:
        df (pd.DataFrame): DataFrame containing target columns.
        target_cols (list): List of target column names.
        load_cached_data (bool): Whether to load from cache if available.
        cache_dir (str): Directory to store the cache file.

    Returns:
        torch.Tensor: Tensor of class weights.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "class_weights.npy")

    if load_cached_data and os.path.exists(cache_path):
        try:
            weights = np.load(cache_path)
            return torch.tensor(weights, dtype=torch.float32)
        except Exception:
            # If loading fails, proceed to recalculate
            pass

    # Calculate weights: n_samples / (n_classes * n_samples_per_class)
    counts = df[target_cols].sum(axis=0).values
    total_samples = counts.sum()
    n_classes = len(target_cols)

    # Add epsilon to avoid division by zero if a class is missing in a subset
    weights = total_samples / (n_classes * (counts + 1e-6))

    # Save to cache
    np.save(cache_path, weights)

    return torch.tensor(weights, dtype=torch.float32)


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates Mean Column-wise ROC AUC.

    Args:
        y_true (np.array or torch.Tensor): Ground truth labels.
        y_pred (np.array or torch.Tensor): Predicted probabilities.

    Returns:
        float: Mean ROC AUC score.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Calculate macro average ROC AUC
    # We use a try-except block because roc_auc_score raises a ValueError
    # if a specific class is not present in y_true (e.g. in small batches)
    try:
        score = roc_auc_score(y_true, y_pred, average="macro", multi_class="ovr")
    except ValueError:
        # Fallback: Calculate per column and average, ignoring columns with 1 class
        scores = []
        num_classes = y_pred.shape[1]
        for i in range(num_classes):
            try:
                # Handle both 1D (indices) and 2D (one-hot) y_true
                if y_true.ndim == 1:
                    y_true_binary = (y_true == i).astype(int)
                else:
                    y_true_binary = y_true[:, i]

                if len(np.unique(y_true_binary)) > 1:
                    s = roc_auc_score(y_true_binary, y_pred[:, i])
                    scores.append(s)
            except ValueError:
                pass

        if len(scores) > 0:
            score = np.mean(scores)
        else:
            score = 0.5  # Default if metric cannot be computed

    return score


def save_checkpoint(model, optimizer, epoch, score, path):
    """
    Saves the model checkpoint.

    Args:
        model: The model to save.
        optimizer: The optimizer.
        epoch (int): Current epoch.
        score (float): Validation score.
        path (str): Path to save the file.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "score": score,
    }
    torch.save(state, path)


def load_checkpoint(model, path, optimizer=None, device="cpu"):
    """
    Loads a model checkpoint.

    Args:
        model: The model to load weights into.
        path (str): Path to the checkpoint file.
        optimizer: Optimizer to load state into (optional).
        device: Device to map location.

    Returns:
        model, optimizer, start_epoch, score
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and checkpoint["optimizer_state_dict"]:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    score = checkpoint.get("score", 0.0)

    return model, optimizer, epoch, score


class AverageMeter:
    """Computes and stores the average and current value."""

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
