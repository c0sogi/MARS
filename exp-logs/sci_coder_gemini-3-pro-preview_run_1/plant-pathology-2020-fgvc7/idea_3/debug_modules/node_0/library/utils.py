import os
import random
import numpy as np
import torch
import pandas as pd


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

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


def get_class_weights(df: pd.DataFrame, target_cols: list) -> torch.Tensor:
    """
    Calculates inverse class frequency weights to handle class imbalance.

    Args:
        df (pd.DataFrame): DataFrame containing the training metadata and target labels.
        target_cols (list): List of column names representing the target classes.

    Returns:
        torch.Tensor: A tensor containing the calculated weight for each class.
    """
    # Calculate the total count for each class
    # Assumes target columns are one-hot encoded or probabilities
    class_counts = df[target_cols].sum().values

    n_samples = len(df)
    n_classes = len(target_cols)

    # Compute inverse class frequency weights
    # Formula: Total_Samples / (Num_Classes * Class_Count)
    # Add a small epsilon to prevent division by zero
    weights = n_samples / (n_classes * (class_counts + 1e-6))

    return torch.tensor(weights, dtype=torch.float32)


def mixup_data(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0, device: torch.device = None
):
    """
    Applies Mixup augmentation to a batch of inputs and targets.

    Args:
        x (torch.Tensor): Batch of input images.
        y (torch.Tensor): Batch of target labels.
        alpha (float): Parameter for the Beta distribution (Beta(alpha, alpha)).
        device (torch.device): The device (CPU/GPU) to use for tensor operations.

    Returns:
        mixed_x (torch.Tensor): The mixed images.
        y_a (torch.Tensor): The original labels.
        y_b (torch.Tensor): The shuffled labels.
        lam (float): The mixing coefficient sampled from the Beta distribution.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    if device:
        index = torch.randperm(batch_size).to(device)
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam
