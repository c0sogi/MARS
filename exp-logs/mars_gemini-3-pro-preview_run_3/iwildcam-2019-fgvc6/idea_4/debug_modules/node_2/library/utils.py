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
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_score(y_true, y_pred):
    """
    Calculates the Macro F1 score for the given predictions.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.

    Returns:
        float: The Macro F1 score.
    """
    return f1_score(y_true, y_pred, average="macro")


def make_weighted_sampler(df, target_col="Category"):
    """
    Creates a WeightedRandomSampler to handle class imbalance by oversampling
    minority classes based on inverse frequency.

    Args:
        df (pd.DataFrame): The training dataframe containing the target column.
        target_col (str): The name of the target column containing class labels.

    Returns:
        WeightedRandomSampler: A sampler configured with weights for each sample.
    """
    # Calculate the count of each class
    class_counts = df[target_col].value_counts()

    # Compute inverse frequency weights: weight = 1.0 / count
    # We use a dictionary for fast lookup: {class_id: weight}
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}

    # Map the class weights to each sample in the dataframe
    # This creates a list of weights corresponding to the order of rows in df
    sample_weights = df[target_col].map(class_weights).values

    # Convert to a DoubleTensor as required by PyTorch for precision
    sample_weights_tensor = torch.from_numpy(sample_weights).double()

    # Create the sampler
    # num_samples equals the dataset length to maintain the epoch size
    # replacement=True is crucial for oversampling
    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True,
    )

    return sampler
