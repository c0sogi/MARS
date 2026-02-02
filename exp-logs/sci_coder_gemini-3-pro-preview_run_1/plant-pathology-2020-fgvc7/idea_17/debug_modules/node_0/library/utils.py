import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

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


def get_class_weights(
    metadata_path: str, load_cached_data: bool = True
) -> torch.Tensor:
    """
    Calculates class weights inversely proportional to class frequencies.
    Implements caching mechanism using .npy format.

    Args:
        metadata_path (str): Path to the training metadata CSV.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        torch.Tensor: A tensor of shape (num_classes,) containing the weights.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "class_weights.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            weights = np.load(cache_path)
            return torch.tensor(weights, dtype=torch.float32).to(Config.DEVICE)
        except Exception as e:
            print(f"Failed to load cached weights: {e}. Recalculating...")

    # 2. Calculate from scratch
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    df = pd.read_csv(metadata_path)

    # Identify target columns based on Config
    target_cols = Config.CLASSES

    # Calculate counts for each class
    # Assuming one-hot or probability encoding in metadata, we sum the columns.
    # If using stratify_label, we could count values, but summing columns is safer for OHE.
    class_counts = df[target_cols].sum().values
    total_samples = len(df)
    num_classes = len(target_cols)

    # Calculate weights: Total / (Num_Classes * Class_Count)
    # This balances the loss contribution of each class.
    weights = total_samples / (num_classes * class_counts)

    # 3. Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    np.save(cache_path, weights)

    return torch.tensor(weights, dtype=torch.float32).to(Config.DEVICE)


def calculate_roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the mean column-wise ROC AUC.

    Args:
        y_true (np.ndarray): Ground truth labels (one-hot encoded).
        y_pred (np.ndarray): Predicted probabilities.

    Returns:
        float: The macro-averaged ROC AUC score.
    """
    # Check if inputs are valid
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    try:
        # average='macro' computes the metric independently for each class and then takes the average
        score = roc_auc_score(y_true, y_pred, average="macro")
        return score
    except ValueError:
        # Handle edge cases where a class might be missing in a small batch/fold (though unlikely in stratified)
        return 0.5


def save_checkpoint(model, path: str, metric_val: float):
    """
    Saves the model state dictionary to the specified path.

    Args:
        model: The PyTorch model.
        path (str): Destination path.
        metric_val (float): The validation metric associated with this checkpoint.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "metric_val": metric_val}, path)
