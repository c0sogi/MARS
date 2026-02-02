import os
import sys
import random
import logging
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name: str = __name__):
    """
    Creates and returns a logger with a standard format.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Prevent adding multiple handlers if logger is already configured
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(console_handler)

    return logger


def calculate_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve.

    Args:
        y_true: Ground truth labels (numpy array or list).
        y_pred: Predicted probabilities (numpy array or list).

    Returns:
        float: The ROC AUC score.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    try:
        # Check if we have more than one class in y_true
        if len(np.unique(y_true)) < 2:
            return 0.5

        score = roc_auc_score(y_true, y_pred)
    except Exception as e:
        print(f"Error calculating AUC: {e}")
        score = 0.5

    return score


def calculate_pos_weight(metadata_path: str):
    """
    Calculates the positive class weight for BCEWithLogitsLoss based on inverse class frequency.
    Formula: pos_weight = number_of_negatives / number_of_positives

    Args:
        metadata_path (str): Path to the training metadata CSV file.

    Returns:
        torch.Tensor: The calculated positive weight wrapped in a tensor.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    df = pd.read_csv(metadata_path)

    if "label" not in df.columns:
        raise ValueError("Metadata CSV must contain a 'label' column.")

    # Count classes
    num_pos = df["label"].sum()
    num_neg = len(df) - num_pos

    if num_pos == 0:
        # Avoid division by zero; if no positives, weight is technically undefined or 0
        # Returning 1.0 implies no weighting
        return torch.tensor([1.0], dtype=torch.float32)

    # Calculate weight: N_neg / N_pos
    # This scales the loss of positive examples up so that the optimizer
    # treats the classes as roughly balanced in terms of total loss contribution.
    weight = num_neg / num_pos

    return torch.tensor([weight], dtype=torch.float32)
