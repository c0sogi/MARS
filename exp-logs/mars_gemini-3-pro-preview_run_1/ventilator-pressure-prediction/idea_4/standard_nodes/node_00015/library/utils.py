import os
import random
import numpy as np
import torch
import hashlib
import json
from typing import Dict, Any


def seed_everything(seed: int = 42) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_cache_hash(config_dict: Dict[str, Any]) -> str:
    """
    Generates a unique MD5 hash based on the provided configuration dictionary.
    This is used to version cached datasets; if the config changes (e.g., features,
    sequence length), the hash changes, invalidating old cache files.

    Args:
        config_dict (dict): A dictionary containing configuration parameters
                            relevant to data processing (e.g., feature list, seq_len).

    Returns:
        str: A hexadecimal MD5 hash string.
    """
    # Filter out keys that might not be serializable or relevant to data structure
    # (though the caller should ideally pass a clean dict)
    # We sort keys to ensure the JSON string is deterministic
    try:
        encoded_config = json.dumps(config_dict, sort_keys=True).encode("utf-8")
    except TypeError as e:
        # Fallback for non-serializable objects: convert to string representation
        # This handles cases where config might contain custom objects
        clean_dict = {k: str(v) for k, v in config_dict.items()}
        encoded_config = json.dumps(clean_dict, sort_keys=True).encode("utf-8")

    return hashlib.md5(encoded_config).hexdigest()


def ensure_dir(file_path: str) -> None:
    """
    Ensures that the directory for a given file path exists.

    Args:
        file_path (str): The full path to a file.
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
