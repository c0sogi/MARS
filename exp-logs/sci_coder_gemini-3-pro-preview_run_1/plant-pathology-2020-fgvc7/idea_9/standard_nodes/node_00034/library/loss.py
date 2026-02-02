import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from library.config import Config


def get_weighted_loss(df, device=Config.DEVICE, load_cached_data=True):
    """
    Constructs a weighted CrossEntropyLoss function to handle class imbalance.
    Calculates class weights based on the provided training DataFrame and caches
    the result to avoid re-computation.

    Args:
        df (pd.DataFrame): The training metadata DataFrame containing target labels.
        device (str): The device (cpu/cuda) where the weight tensor should be stored.
        load_cached_data (bool): If True, attempts to load pre-calculated weights from disk.

    Returns:
        nn.CrossEntropyLoss: The configured loss function with class weights.
    """
    # Ensure the working directory exists for caching
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "class_weights.npy")

    weights = None

    # 1. Attempt to load cached weights
    if load_cached_data and os.path.exists(cache_path):
        try:
            weights = np.load(cache_path)
        except Exception:
            # If loading fails (e.g., corrupt file), force recalculation
            weights = None

    # 2. Calculate weights if not cached or loading failed
    if weights is None:
        # Retrieve class labels from configuration
        class_labels = Config.CLASS_LABELS

        # Determine the class label for each sample in the DataFrame
        # Assumes columns in df correspond to class_labels
        # We use idxmax to find the dominant class (ground truth)
        y_classes = df[class_labels].idxmax(axis=1)

        # Count the frequency of each class
        class_counts = y_classes.value_counts().to_dict()

        total_samples = len(df)
        num_classes = len(class_labels)

        weights_list = []

        # Compute weight for each class based on the order in CLASS_LABELS
        for label in class_labels:
            count = class_counts.get(label, 0)
            if count > 0:
                # Balanced Weight Formula: Total / (Num_Classes * Count)
                # This upweights minority classes and downweights majority classes
                w = total_samples / (num_classes * count)
            else:
                # Fallback for classes with 0 samples (e.g., in small debug subsets)
                w = 1.0
            weights_list.append(w)

        weights = np.array(weights_list, dtype=np.float32)

        # Save the calculated weights to cache
        np.save(cache_path, weights)

    # 3. Instantiate and return the weighted loss function
    # Convert numpy array to torch tensor and move to the specified device
    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    return criterion
