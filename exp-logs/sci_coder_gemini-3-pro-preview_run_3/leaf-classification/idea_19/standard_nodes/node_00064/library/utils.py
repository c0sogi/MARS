import os
import random
import numpy as np
import torch
import joblib
import pandas as pd
from library.config import Config


def seed_everything(seed: int = Config.SEED) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clip_probabilities(probs: np.ndarray) -> np.ndarray:
    """
    Clips probabilities to the range [EPSILON, 1 - EPSILON] to avoid log loss extremes.

    Args:
        probs: Numpy array of probabilities.

    Returns:
        Numpy array with clipped probabilities.
    """
    return np.clip(probs, Config.CLIP_MIN, Config.CLIP_MAX)


def ensure_directory(path: str) -> None:
    """
    Ensures that the directory for the given file path exists.

    Args:
        path: The file path for which to check/create the parent directory.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def save_numpy(array: np.ndarray, path: str) -> None:
    """
    Saves a NumPy array to a file, ensuring the directory exists.

    Args:
        array: The numpy array to save.
        path: The destination file path.
    """
    ensure_directory(path)
    np.save(path, array)


def load_numpy(path: str) -> np.ndarray:
    """
    Loads a NumPy array from a file.

    Args:
        path: The source file path.

    Returns:
        The loaded numpy array.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return np.load(path, allow_pickle=True)


def save_pickle(obj, path: str) -> None:
    """
    Saves a Python object (e.g., model pipeline) using joblib, ensuring the directory exists.

    Args:
        obj: The object to save.
        path: The destination file path.
    """
    ensure_directory(path)
    joblib.dump(obj, path)


def load_pickle(path: str):
    """
    Loads a Python object using joblib.

    Args:
        path: The source file path.

    Returns:
        The loaded Python object.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return joblib.load(path)


def save_submission(submission_df: pd.DataFrame, path: str = None) -> None:
    """
    Saves the submission DataFrame to a CSV file.

    Args:
        submission_df: The pandas DataFrame containing the submission.
        path: The destination file path. Defaults to Config.SUBMISSION_PATH.
    """
    if path is None:
        path = Config.SUBMISSION_PATH
    ensure_directory(path)
    submission_df.to_csv(path, index=False)
