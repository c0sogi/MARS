import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

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


def get_device() -> torch.device:
    """
    Returns the appropriate PyTorch device (CUDA or CPU).

    Returns:
        torch.device: The device object.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(state: dict, filepath: str):
    """
    Saves a model checkpoint to the specified filepath.
    Ensures the directory exists before saving.

    Args:
        state (dict): The state dictionary containing model parameters, optimizer state, etc.
        filepath (str): The path where the checkpoint will be saved.
    """
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(filepath: str, device: torch.device = None) -> dict:
    """
    Loads a model checkpoint from the specified filepath.

    Args:
        filepath (str): The path to the checkpoint file.
        device (torch.device, optional): The device to map the location to.

    Returns:
        dict: The loaded state dictionary.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found at {filepath}")

    if device is None:
        device = get_device()

    return torch.load(filepath, map_location=device)


def print_metrics(metrics: dict, prefix: str = ""):
    """
    Prints metric values with full precision.

    Args:
        metrics (dict): A dictionary of metric names and values.
        prefix (str): An optional prefix string to print before the metrics.
    """
    message_parts = []
    if prefix:
        message_parts.append(prefix)

    for k, v in metrics.items():
        message_parts.append(f"{k}: {v}")

    print(" | ".join(message_parts))


def save_npy(data: np.ndarray, filepath: str):
    """
    Saves a numpy array to a file, ensuring the directory exists.

    Args:
        data (np.ndarray): The data to save.
        filepath (str): The destination path.
    """
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    np.save(filepath, data)


def load_npy(filepath: str) -> np.ndarray:
    """
    Loads a numpy array from a file.

    Args:
        filepath (str): The path to the .npy file.

    Returns:
        np.ndarray: The loaded data.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return np.load(filepath)


def save_parquet(df: pd.DataFrame, filepath: str):
    """
    Saves a pandas DataFrame to a parquet file, ensuring the directory exists.

    Args:
        df (pd.DataFrame): The dataframe to save.
        filepath (str): The destination path.
    """
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    df.to_parquet(filepath, index=False)


def load_parquet(filepath: str) -> pd.DataFrame:
    """
    Loads a pandas DataFrame from a parquet file.

    Args:
        filepath (str): The path to the .parquet file.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_parquet(filepath)


def format_time(seconds: float) -> str:
    """
    Formats a duration in seconds to a string (hh:mm:ss).

    Args:
        seconds (float): Duration in seconds.

    Returns:
        str: Formatted time string.
    """
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
