import os
import random
import numpy as np
import torch
import hashlib
import json


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The random seed value to set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cache_hash(feature_names: list, config_dict: dict = None) -> str:
    """
    Generates a unique MD5 hash based on the feature list and configuration dictionary.
    This hash is used to create unique filenames for cached datasets to prevent
    stale data artifacts when configuration or features change.

    Args:
        feature_names (list): List of feature names used in the dataset.
        config_dict (dict, optional): Dictionary of configuration parameters
                                      (e.g., SEQ_LEN, preprocessing constants).

    Returns:
        str: A hexadecimal MD5 hash string representing the inputs.
    """
    # Ensure feature list is sorted for consistency
    sorted_features = sorted(feature_names) if feature_names else []

    # Base content to hash
    hash_content = {"features": sorted_features}

    # Incorporate configuration if provided
    if config_dict:
        clean_config = {}
        # Iterate through config to ensure values are serializable
        for k, v in config_dict.items():
            try:
                # Check if value is JSON serializable
                json.dumps(v)
                clean_config[k] = v
            except (TypeError, OverflowError):
                # Fallback for non-serializable types (like functions or classes)
                if hasattr(v, "__name__"):
                    clean_config[k] = v.__name__
                else:
                    clean_config[k] = str(v)

        # Sort keys in the config dictionary for determinism
        hash_content["config"] = clean_config

    # Serialize to JSON with sort_keys=True to ensure deterministic string representation
    serialized_content = json.dumps(hash_content, sort_keys=True)

    # Compute MD5 hash
    hash_object = hashlib.md5(serialized_content.encode("utf-8"))
    return hash_object.hexdigest()
