import os
import random
import hashlib
import json
import numpy as np
import torch


def set_seed(seed: int = 42):
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


def get_config_hash(config_dict: dict, input_files: list = None) -> str:
    """
    Generates a unique hash based on the configuration dictionary and input file sizes.
    This ensures that if data changes or config changes, the hash changes, invalidating old caches.

    Args:
        config_dict: Dictionary containing configuration parameters.
        input_files: List of file paths to check for size changes.

    Returns:
        A 10-character MD5 hash string.
    """
    # Create a copy to avoid modifying the original dict
    conf_copy = config_dict.copy()

    # Serialize config with sorted keys for consistency
    # Using default=str to handle non-serializable objects gracefully
    config_str = json.dumps(conf_copy, sort_keys=True, default=str)

    file_info = ""
    if input_files:
        # Sort files to ensure list order doesn't affect hash
        for file_path in sorted(input_files):
            if os.path.exists(file_path):
                # Get file size to detect data changes
                size = os.path.getsize(file_path)
                file_info += f"{file_path}:{size}|"
            else:
                # If file is missing, include that state in hash
                file_info += f"{file_path}:MISSING|"

    # Combine config and file info
    payload = config_str + "||" + file_info

    # Compute MD5
    hash_object = hashlib.md5(payload.encode("utf-8"))

    # Return first 10 characters
    return hash_object.hexdigest()[:10]
