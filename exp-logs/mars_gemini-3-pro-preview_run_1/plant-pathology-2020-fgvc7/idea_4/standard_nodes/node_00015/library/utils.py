import os
import random
import numpy as np
import torch
import pandas as pd


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_class_weights(
    df: pd.DataFrame, target_cols: list, device: str = "cpu"
) -> torch.Tensor:
    """
    Calculates inverse class frequency weights for Weighted Cross-Entropy Loss.

    Formula: n_samples / (n_classes * n_samples_j)

    Args:
        df (pd.DataFrame): The dataframe containing training labels.
        target_cols (list): List of column names corresponding to the target classes.
        device (str): The device ('cpu' or 'cuda') to move the weights tensor to.

    Returns:
        torch.Tensor: A tensor of shape (num_classes,) containing the calculated weights.
    """
    # Calculate the count for each class
    # We assume the target columns in the dataframe are one-hot encoded or
    # represent probabilities where the sum roughly equals the count.
    class_counts = df[target_cols].sum().values

    total_samples = len(df)
    num_classes = len(target_cols)

    # Handle potential zero counts to avoid division by zero (though unlikely in this dataset)
    class_counts = np.maximum(class_counts, 1)

    # Calculate weights: Total / (Num_Classes * Class_Count)
    # This scales the loss such that each class contributes equally on average
    weights = total_samples / (num_classes * class_counts)

    # Convert to tensor and move to the specified device
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    return weights_tensor
