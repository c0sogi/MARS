import os
import random
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_spearman(y_true, y_pred, target_cols=None):
    """
    Computes the mean column-wise Spearman's correlation coefficient.

    Args:
        y_true (np.ndarray or pd.DataFrame): Ground truth values.
        y_pred (np.ndarray or pd.DataFrame): Predicted probabilities.
        target_cols (list, optional): List of column names for logging/debugging.

    Returns:
        float: The mean Spearman's rank correlation coefficient across all columns.
    """
    # Convert inputs to numpy arrays if they are DataFrames
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    n_cols = y_true.shape[1]
    corrs = []

    for i in range(n_cols):
        col_true = y_true[:, i]
        col_pred = y_pred[:, i]

        # Check for constant values which cause spearmanr to return NaN or warn
        if np.std(col_pred) == 0 or np.std(col_true) == 0:
            corr = 0.0
        else:
            # spearmanr returns (correlation, p-value)
            corr, _ = spearmanr(col_true, col_pred)

        # Handle NaN result if it occurs despite checks
        if np.isnan(corr):
            corr = 0.0

        corrs.append(corr)

    return np.mean(corrs)


def get_artifact_path(filename):
    """
    Resolves the full path for an artifact within the configured working directory.
    Creates the directory if it does not exist.

    Args:
        filename (str): Name of the file.

    Returns:
        str: Full path to the file.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    return os.path.join(Config.WORKING_DIR, filename)


def save_numpy_array(array, filename):
    """
    Saves a numpy array to the working directory.

    Args:
        array (np.ndarray): The array to save.
        filename (str): The filename (should end with .npy).
    """
    path = get_artifact_path(filename)
    np.save(path, array)


def load_numpy_array(filename):
    """
    Loads a numpy array from the working directory if it exists.

    Args:
        filename (str): The filename (should end with .npy).

    Returns:
        np.ndarray or None: The loaded array, or None if file not found.
    """
    path = get_artifact_path(filename)
    if os.path.exists(path):
        return np.load(path)
    return None


def save_dataframe(df, filename):
    """
    Saves a pandas DataFrame to the working directory using Parquet format.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): The filename (should end with .parquet).
    """
    path = get_artifact_path(filename)
    df.to_parquet(path, index=False)


def load_dataframe(filename):
    """
    Loads a pandas DataFrame from the working directory.

    Args:
        filename (str): The filename (should end with .parquet).

    Returns:
        pd.DataFrame or None: The loaded DataFrame, or None if file not found.
    """
    path = get_artifact_path(filename)
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


def save_checkpoint(state_dict, filename):
    """
    Saves a PyTorch model state dictionary.

    Args:
        state_dict (dict): The model state dict.
        filename (str): The filename (e.g., model.pth).
    """
    path = get_artifact_path(filename)
    torch.save(state_dict, path)


def load_checkpoint(filename, device="cpu"):
    """
    Loads a PyTorch model state dictionary.

    Args:
        filename (str): The filename (e.g., model.pth).
        device (str or torch.device): Device to map the model to.

    Returns:
        dict or None: The loaded state dict, or None if file not found.
    """
    path = get_artifact_path(filename)
    if os.path.exists(path):
        return torch.load(path, map_location=device)
    return None
