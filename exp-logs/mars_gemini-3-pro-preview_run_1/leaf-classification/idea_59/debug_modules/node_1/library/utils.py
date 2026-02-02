import os
import random
import hashlib
import json
import numpy as np
import torch


def set_seed(seed):
    """
    Initializes random number generators for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The random seed to set.
    """
    # Python std lib
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Enforce deterministic behavior for cuDNN to ensure consistent results
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_config_hash(config):
    """
    Generates a unique identifier (MD5 hash) for a specific configuration dictionary.
    This is used to create unique cache filenames based on active feature flags and parameters.

    Args:
        config (dict): The configuration dictionary to hash.

    Returns:
        str: The hexadecimal MD5 hash string.
    """
    # Sort keys to ensure the hash is deterministic regardless of dictionary insertion order
    # Use default=str to handle types that are not natively JSON serializable (e.g., numpy types, sets)
    config_str = json.dumps(config, sort_keys=True, default=str)

    # Return MD5 hash
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()
