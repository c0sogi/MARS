import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import log_loss
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metric(y_true, y_pred):
    """
    Computes the Multi-class Logarithmic Loss.

    Predicted probabilities are clipped to [1e-15, 1-1e-15] before scoring
    to avoid extremes of the log function.

    Args:
        y_true: Array-like of ground truth labels (indices or one-hot).
        y_pred: Array-like of predicted probabilities.

    Returns:
        float: The log loss value.
    """
    # Epsilon value as defined in the task metric
    eps = 1e-15

    # Clip probabilities
    # Note: sklearn log_loss does this internally with the eps parameter,
    # but we ensure it matches the spec exactly.
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # Calculate log loss
    # labels parameter ensures that even if a class is missing in a batch,
    # the loss is calculated correctly with respect to the global classes.
    # However, y_true here is expected to be numeric indices or strings matching classes.
    # Assuming y_true matches the format expected by sklearn (labels or indices).
    score = log_loss(y_true, y_pred)

    return score


def save_artifact(data, filename: str):
    """
    Saves an intermediate artifact (DataFrame or Numpy array) to the working directory.
    Does NOT use pickle.

    Args:
        data: The pandas DataFrame or numpy array to save.
        filename (str): The name of the file (e.g., 'features.npy', 'data.parquet').
    """
    file_path = os.path.join(Config.WORKING_DIR, filename)

    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if isinstance(data, pd.DataFrame):
        if not filename.endswith(".parquet"):
            raise ValueError("Pandas DataFrames must be saved with .parquet extension.")
        data.to_parquet(file_path, index=False)

    elif isinstance(data, np.ndarray):
        if not filename.endswith(".npy"):
            raise ValueError("Numpy arrays must be saved with .npy extension.")
        np.save(file_path, data)

    else:
        raise TypeError(
            f"Unsupported data type for saving: {type(data)}. Only pd.DataFrame and np.ndarray are supported."
        )


def load_artifact(filename: str):
    """
    Loads an intermediate artifact from the working directory.

    Args:
        filename (str): The name of the file to load.

    Returns:
        The loaded data (pd.DataFrame or np.ndarray), or None if file does not exist.
    """
    file_path = os.path.join(Config.WORKING_DIR, filename)

    if not os.path.exists(file_path):
        return None

    if filename.endswith(".parquet"):
        return pd.read_parquet(file_path)
    elif filename.endswith(".npy"):
        return np.load(file_path)
    else:
        raise ValueError(
            f"Unsupported file extension for loading: {filename}. Only .parquet and .npy are supported."
        )
