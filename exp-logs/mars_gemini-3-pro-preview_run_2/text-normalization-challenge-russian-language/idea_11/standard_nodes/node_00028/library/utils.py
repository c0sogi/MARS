import os
import sys
import time
import random
import gc
import numpy as np
import torch
import pandas as pd
from contextlib import contextmanager
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    """
    Returns the PyTorch device configured in Config.

    Returns:
        torch.device: The device (cpu or cuda).
    """
    return torch.device(Config.DEVICE)


def cleanup():
    """
    Clears memory by running garbage collection and emptying the CUDA cache.
    Useful to call between major processing steps or training epochs.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def timer(name):
    """
    Context manager to measure and print the execution time of a code block.

    Args:
        name (str): The name of the operation being timed.
    """
    t0 = time.time()
    yield
    elapsed = time.time() - t0
    print(f"[{name}] done in {elapsed:.2f} s")


def load_raw_data(split):
    """
    Loads the raw dataset for the specified split using paths defined in Config.

    Args:
        split (str): One of 'train', 'val', or 'test'.

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
        raise ValueError(f"Unknown split: {split}. Must be 'train', 'val', or 'test'.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"{split} data not found at {path}")

    # Load using pandas
    # We enforce string types for text columns to handle tokens that might look like NaNs or numbers
    dtype_dict = {"before": str, "sentence_id": str}

    # 'after' and 'class' columns exist only in train/val
    if split != "test":
        dtype_dict["after"] = str
        dtype_dict["class"] = str

    df = pd.read_csv(path, dtype=dtype_dict)

    # Fill any residual NaNs with empty strings (though metadata script should have handled this)
    df.fillna("", inplace=True)

    return df


def sample_dataset(df, n_samples, seed=42):
    """
    Samples a subset of the dataframe for debugging purposes if n_samples > 0.

    Args:
        df (pd.DataFrame): The dataframe to sample.
        n_samples (int): Number of samples to keep. If 0, returns full dataframe.
        seed (int): Random seed for sampling.

    Returns:
        pd.DataFrame: The sampled dataframe.
    """
    if n_samples > 0 and n_samples < len(df):
        print(f"Subsetting dataset from {len(df)} to {n_samples} samples.")
        return df.sample(n=n_samples, random_state=seed).reset_index(drop=True)
    return df


def save_to_cache(data, path):
    """
    Saves data to the specified path, ensuring the directory exists.
    Supports pandas DataFrame (.parquet) and numpy arrays (.npy).
    Strictly avoids pickle for data storage.

    Args:
        data: The data object (DataFrame or ndarray).
        path (str): The destination file path.
    """
    # Ensure directory exists
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    if path.endswith(".parquet"):
        if isinstance(data, pd.DataFrame):
            data.to_parquet(path, index=False)
        else:
            raise ValueError("Data must be a pandas DataFrame for .parquet extension")
    elif path.endswith(".npy"):
        if isinstance(data, np.ndarray):
            np.save(path, data)
        else:
            raise ValueError("Data must be a numpy array for .npy extension")
    else:
        # We do not support pickle for data caching
        raise ValueError(
            f"Unsupported file extension for caching: {path}. Use .parquet or .npy"
        )


def load_from_cache(path):
    """
    Loads data from the specified path if it exists.

    Args:
        path (str): The file path to load.

    Returns:
        The loaded data (DataFrame or ndarray) or None if file does not exist.
    """
    if not os.path.exists(path):
        return None

    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.endswith(".npy"):
        # allow_pickle=False ensures we strictly load numerical/structured data
        return np.load(path, allow_pickle=False)
    else:
        raise ValueError(f"Unsupported file extension for loading: {path}")


def print_metrics(metrics):
    """
    Prints metrics dictionary with full precision.

    Args:
        metrics (dict): Dictionary of metric names and values.
    """
    print("Validation Metrics:")
    for k, v in metrics.items():
        # Print full precision as requested
        print(f"{k}: {v}")
