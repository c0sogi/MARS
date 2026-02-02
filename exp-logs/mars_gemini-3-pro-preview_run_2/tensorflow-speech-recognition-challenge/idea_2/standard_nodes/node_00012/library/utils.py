import torch
import numpy as np
import pandas as pd
from torch.utils.data import WeightedRandomSampler
from library.config import set_seed


def get_weighted_sampler(
    df: pd.DataFrame, label_col: str = "label"
) -> WeightedRandomSampler:
    """
    Creates a WeightedRandomSampler to handle class imbalance in the dataset.

    It calculates the inverse frequency of each class and assigns a corresponding
    weight to each sample. This ensures that when used with a DataLoader,
    each batch contains a balanced representation of all classes.

    Args:
        df (pd.DataFrame): The metadata DataFrame containing the dataset information.
        label_col (str): The name of the column containing the class labels.

    Returns:
        WeightedRandomSampler: A sampler configured with sample weights.
    """
    # Extract labels from the dataframe
    labels = df[label_col].values

    # Count the frequency of each class
    classes, counts = np.unique(labels, return_counts=True)

    # Calculate inverse frequency weights: weight = 1.0 / count
    class_weights = {cls: 1.0 / count for cls, count in zip(classes, counts)}

    # Assign a weight to each sample based on its label
    sample_weights = [class_weights[label] for label in labels]

    # Convert weights to a torch DoubleTensor (recommended for precision in sampling)
    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.double)

    # Create the WeightedRandomSampler
    # num_samples is set to len(df) to maintain the epoch size
    # replacement=True allows oversampling of minority classes
    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True,
    )

    return sampler
