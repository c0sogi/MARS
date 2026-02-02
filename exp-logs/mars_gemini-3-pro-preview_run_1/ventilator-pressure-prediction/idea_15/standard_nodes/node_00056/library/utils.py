import os
import random
import numpy as np
import torch
import hashlib
import json


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

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


def get_cache_hash(config_dict):
    """
    Generates a unique MD5 hash based on a configuration dictionary.
    This is used to version cache files based on features and hyperparameters.

    Args:
        config_dict (dict): Dictionary containing configuration parameters
                            (e.g., feature list, sequence length).
                            Must be JSON-serializable.

    Returns:
        str: A hexadecimal MD5 hash string.
    """
    # Sort keys to ensure consistent ordering for hashing regardless of dict insertion order
    serialized = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()


def get_cache_path(directory, prefix, config_hash, extension=".parquet"):
    """
    Constructs a file path for a cached dataset.

    Args:
        directory (str): The directory where the cache file should be stored.
        prefix (str): A prefix for the filename (e.g., 'train', 'val').
        config_hash (str): The hash string representing the configuration.
        extension (str): The file extension (default: .parquet).

    Returns:
        str: The full path to the cache file.
    """
    filename = f"{prefix}_{config_hash}{extension}"
    return os.path.join(directory, filename)


def ensure_dir(file_path):
    """
    Ensures that the directory for the given file path exists.

    Args:
        file_path (str): The full path to a file.
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
