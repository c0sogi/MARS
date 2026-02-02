import os
import random
import numpy as np
import pandas as pd
import torch
import warnings
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """
    Returns the PyTorch device (CUDA if available, else CPU).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def clean_text(text_series):
    """
    Preprocesses a pandas Series of text.
    - Handles NaNs by replacing them with empty strings.
    - Ensures all entries are strings.
    - Normalizes whitespace (replaces newlines/tabs with spaces).

    Args:
        text_series (pd.Series): Input text data.

    Returns:
        pd.Series: Cleaned text data.
    """
    # Fill NaNs with empty string
    cleaned = text_series.fillna("")

    # Ensure string type
    cleaned = cleaned.astype(str)

    # Normalize whitespace: replace newlines and tabs with single space
    # and strip leading/trailing whitespace
    cleaned = cleaned.apply(lambda x: " ".join(x.split()))

    return cleaned


def load_data(split, debug=Config.DEBUG):
    """
    Loads the requested dataset split from the metadata directory.

    Args:
        split (str): One of 'train', 'val', 'test'.
        debug (bool): If True, returns a subsample of the data for debugging.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    if split == "train":
        path = Config.TRAIN_DATA_PATH
    elif split == "val":
        path = Config.VAL_DATA_PATH
    elif split == "test":
        path = Config.TEST_DATA_PATH
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}")

    df = pd.read_parquet(path)

    if debug:
        # Sample a small fraction for debugging purposes
        # Using a fixed seed for reproducibility even in debug mode
        sample_size = min(100, len(df))
        df = df.sample(n=sample_size, random_state=Config.SEED).reset_index(drop=True)
        # print(f"DEBUG MODE: Loaded {len(df)} samples from {split} split.")

    return df


def save_to_cache(data, filename):
    """
    Saves data to the working directory cache.
    Supports .npy, .npz, and .parquet extensions.

    Args:
        data: The data object to save (numpy array, sparse matrix, or DataFrame).
        filename (str): The filename (including extension).
    """
    filepath = os.path.join(Config.WORKING_DIR, filename)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    if filename.endswith(".npy"):
        np.save(filepath, data)
    elif filename.endswith(".npz"):
        import scipy.sparse

        if scipy.sparse.issparse(data):
            scipy.sparse.save_npz(filepath, data)
        else:
            np.savez_compressed(filepath, data=data)
    elif filename.endswith(".parquet"):
        if isinstance(data, pd.DataFrame):
            data.to_parquet(filepath, index=False)
        else:
            raise ValueError("Data must be a pandas DataFrame for .parquet saving.")
    else:
        raise ValueError(f"Unsupported file extension for caching: {filename}")


def load_from_cache(filename):
    """
    Loads data from the working directory cache.

    Args:
        filename (str): The filename to load.

    Returns:
        The loaded data object, or None if file does not exist.
    """
    filepath = os.path.join(Config.WORKING_DIR, filename)

    if not os.path.exists(filepath):
        return None

    if filename.endswith(".npy"):
        return np.load(filepath)
    elif filename.endswith(".npz"):
        import scipy.sparse

        # Try loading as sparse matrix first
        try:
            return scipy.sparse.load_npz(filepath)
        except:
            # Fallback to standard numpy archive
            with np.load(filepath) as data:
                return data["data"]
    elif filename.endswith(".parquet"):
        return pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported file extension for loading: {filename}")
