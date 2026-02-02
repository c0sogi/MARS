import json
import hashlib
import torch
from library.config import Config


def set_seed(seed: int = Config.SEED) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    Delegates to the centralized Config.set_seed method.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    Config.set_seed(seed)


def get_device() -> torch.device:
    """
    Determines and returns the PyTorch device to be used for computation.

    Returns:
        torch.device: The device specified in Config.DEVICE (e.g., 'cuda' or 'cpu').
    """
    return torch.device(Config.DEVICE)


def generate_config_hash(config_dict: dict = None) -> str:
    """
    Generates a unique MD5 hash based on a configuration dictionary.
    This hash is used to create unique filenames for cached datasets, ensuring
    that changes in configuration (like feature lists or sequence lengths)
    invalidate old caches.

    Args:
        config_dict (dict, optional): A dictionary of configuration parameters.
                                      If None, retrieves the default data hash config
                                      from Config.get_data_hash_config().

    Returns:
        str: The hexadecimal MD5 hash string.
    """
    if config_dict is None:
        config_dict = Config.get_data_hash_config()

    # Use json.dumps with sort_keys=True to ensure deterministic string representation
    # of the dictionary, regardless of key insertion order.
    config_str = json.dumps(config_dict, sort_keys=True)

    # Generate MD5 hash
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()
