import os
import random
import numpy as np
import torch
import pandas as pd


def seed_everything(seed=42):
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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_pos_weights(df, device):
    """
    Calculates positive class weights for BCEWithLogitsLoss based on class imbalance.
    Formula: pos_weight = number_of_negatives / number_of_positives

    Args:
        df (pd.DataFrame): DataFrame containing the training labels.
                           Columns starting with 'species_' are considered targets.
        device (torch.device): The device to place the weights tensor on.

    Returns:
        torch.Tensor: A tensor of weights with shape (num_classes,).
    """
    # Identify label columns
    label_cols = [c for c in df.columns if c.startswith("species_")]

    # Extract labels as numpy array
    labels = df[label_cols].values

    # Calculate counts
    num_samples = len(labels)
    pos_counts = np.sum(labels, axis=0)
    neg_counts = num_samples - pos_counts

    # Calculate weights: neg / pos
    # Add a small epsilon to avoid division by zero if a class has 0 samples (unlikely but safe)
    pos_weights = neg_counts / (pos_counts + 1e-6)

    return torch.as_tensor(pos_weights, dtype=torch.float32, device=device)
