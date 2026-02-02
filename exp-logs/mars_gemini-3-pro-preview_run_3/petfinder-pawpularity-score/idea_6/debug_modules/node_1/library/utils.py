import os
import random
import hashlib
import json
import numpy as np
import torch


def set_seed(seed=42):
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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_config_hash(config):
    """
    Generates a unique MD5 hash based on the critical configuration parameters
    that affect feature extraction. This ensures that if backbones, image size,
    or TTA settings change, a new cache file is generated.

    Args:
        config: The Config class or object containing settings.

    Returns:
        str: A hexadecimal hash string representing the configuration state.
    """
    # Extract settings that directly impact the generated feature values
    relevant_settings = {
        "backbones": config.BACKBONES,
        "image_size": config.IMAGE_SIZE,
        "use_tta": config.USE_TTA,
    }

    # Serialize to JSON with sort_keys=True to ensure deterministic string representation
    settings_str = json.dumps(relevant_settings, sort_keys=True)

    # Generate MD5 hash
    config_hash = hashlib.md5(settings_str.encode("utf-8")).hexdigest()
    return config_hash


def save_cache(data, filepath):
    """
    Saves a NumPy array to the specified filepath using .npy format.
    Automatically creates the parent directory if it does not exist.

    Args:
        data (np.ndarray): The numpy array to save.
        filepath (str): The full destination path (should end in .npy).
    """
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Save using standard numpy binary format
    np.save(filepath, data)


def load_cache(filepath):
    """
    Loads a NumPy array from the specified filepath if it exists.

    Args:
        filepath (str): The full path to the .npy file.

    Returns:
        np.ndarray or None: The loaded data if successful, else None.
    """
    if os.path.exists(filepath):
        try:
            # allow_pickle=False ensures we only load pure array data, consistent with requirements
            return np.load(filepath, allow_pickle=False)
        except Exception as e:
            print(f"Error loading cache from {filepath}: {e}")
            return None
    return None
