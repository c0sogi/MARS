import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed: int = Config.SEED) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_weights(df: pd.DataFrame, label_col: str = "label") -> float:
    """
    Calculates the inverse class frequency weight for the positive class.
    This is used as the 'pos_weight' argument in BCEWithLogitsLoss to handle class imbalance.

    Formula: number_of_negatives / number_of_positives

    Args:
        df (pd.DataFrame): The dataframe containing the training metadata.
        label_col (str): The name of the column containing class labels (0 or 1).

    Returns:
        float: The calculated positive weight.
    """
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found in DataFrame.")

    # Count occurrences of each class
    counts = df[label_col].value_counts()

    neg_count = counts.get(0, 0)
    pos_count = counts.get(1, 0)

    if pos_count == 0:
        # Prevent division by zero if no positive samples exist
        return 1.0

    weight = neg_count / pos_count
    return weight
