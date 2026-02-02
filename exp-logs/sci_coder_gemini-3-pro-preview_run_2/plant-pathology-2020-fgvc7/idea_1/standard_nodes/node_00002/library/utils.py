import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_class_weights(
    train_df, target_cols=Config.TARGET_COLS, device=Config.DEVICE
):
    """
    Computes inverse class frequency weights for the loss function to handle class imbalance.
    Formula: weight_class = total_samples / (num_classes * samples_in_class)

    Args:
        train_df (pd.DataFrame): The training metadata DataFrame containing one-hot encoded labels.
        target_cols (list): List of column names corresponding to the target classes.
        device (torch.device): The device (CPU/GPU) to place the weight tensor on.

    Returns:
        torch.Tensor: A tensor of weights with shape (num_classes,).
    """
    # Calculate the number of samples for each class
    # Assumes target columns are one-hot encoded or binary indicators
    class_counts = train_df[target_cols].sum().values

    # Total number of samples in the dataset
    total_samples = class_counts.sum()

    # Number of classes
    num_classes = len(target_cols)

    # Compute weights
    # Adding a small epsilon is not strictly necessary if we are sure counts > 0,
    # but the formula handles it naturally.
    # We use float division.
    weights = total_samples / (num_classes * class_counts)

    # Convert to PyTorch tensor and move to the specified device
    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    return weight_tensor
