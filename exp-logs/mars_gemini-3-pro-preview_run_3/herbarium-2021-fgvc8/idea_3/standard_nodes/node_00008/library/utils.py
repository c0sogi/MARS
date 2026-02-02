import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import f1_score
from torch.utils.data import WeightedRandomSampler
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_f1_macro(y_true, y_pred):
    """
    Calculates the Macro F1 score.

    Args:
        y_true (np.array or list): Ground truth labels.
        y_pred (np.array or list): Predicted labels.

    Returns:
        float: Macro F1 score.
    """
    return f1_score(y_true, y_pred, average="macro")


def get_weighted_sampler(df, load_cached_data=True):
    """
    Creates a WeightedRandomSampler to handle class imbalance.
    Implements caching for the calculated weights to speed up subsequent runs.

    Args:
        df (pd.DataFrame): DataFrame containing a 'category_id' column.
        load_cached_data (bool): Whether to try loading weights from cache.

    Returns:
        WeightedRandomSampler: Sampler initialized with inverse frequency weights.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache path
    # Differentiate cache for debug mode to avoid size mismatches
    suffix = "_debug" if Config.DEBUG else ""
    cache_filename = f"train_sample_weights{suffix}.npy"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    weights = None

    # 1. Try to load from cache
    if load_cached_data:
        if os.path.exists(cache_path):
            try:
                loaded_weights = np.load(cache_path)
                # Verify length matches dataframe to ensure cache validity
                if len(loaded_weights) == len(df):
                    weights = loaded_weights
            except Exception:
                # If loading fails for any reason, we will recompute
                weights = None

    # 2. Compute if not loaded
    if weights is None:
        # Calculate class counts
        # df['category_id'] is expected to contain the class labels
        class_counts = df["category_id"].value_counts().sort_index()

        # Calculate weight for each class: 1 / count
        class_weights = 1.0 / class_counts

        # Map weights to each sample in the dataframe
        # We convert to dict for efficient mapping
        class_weights_dict = class_weights.to_dict()
        weights = df["category_id"].map(class_weights_dict).values.astype(np.float64)

        # Save the computed weights to cache
        np.save(cache_path, weights)

    # Convert numpy array to torch tensor
    weights_tensor = torch.from_numpy(weights).float()

    # Create the sampler
    # num_samples is set to the length of the dataset to maintain epoch size
    sampler = WeightedRandomSampler(
        weights=weights_tensor, num_samples=len(weights_tensor), replacement=True
    )

    return sampler
