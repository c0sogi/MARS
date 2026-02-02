import os
import random
import numpy as np
import torch
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


def get_class_weights(df, target_cols, load_cached_data=True):
    """
    Computes or loads inverse frequency class weights to handle class imbalance.

    Args:
        df (pd.DataFrame): The training dataframe.
        target_cols (list): List of target column names.
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        torch.Tensor: Class weights on the configured device.
    """
    cache_path = Config.get_cache_path("class_weights.npy")

    # Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            weights = np.load(cache_path)
            return torch.tensor(weights, dtype=torch.float32).to(Config.DEVICE)
        except Exception:
            # If load fails, proceed to compute
            pass

    # Compute weights: w_j = n_samples / (n_classes * n_samples_j)
    total_samples = len(df)
    num_classes = len(target_cols)

    # Calculate count of positive samples for each class
    class_counts = df[target_cols].sum(axis=0).values

    # Prevent division by zero (though unlikely in this dataset)
    class_counts = np.maximum(class_counts, 1)

    weights = total_samples / (num_classes * class_counts)

    # Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    np.save(cache_path, weights)

    return torch.tensor(weights, dtype=torch.float32).to(Config.DEVICE)


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the Mean Column-wise ROC AUC.

    Args:
        y_true: Ground truth labels (N, num_classes). Can be np.array or torch.Tensor.
        y_pred: Predicted probabilities (N, num_classes). Can be np.array or torch.Tensor.

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
        # Handle edge cases (e.g., only one class present in batch)
        score = 0.0

    return score
