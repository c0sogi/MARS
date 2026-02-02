import os
import random
import numpy as np
import hashlib
import json
from library.config import SEED


def set_seed(seed: int = SEED):
    """
    Sets the random seed for reproducibility across Python's random module,
    NumPy, and the OS environment variable PYTHONHASHSEED.

    Args:
        seed (int): The seed value to use. Defaults to the global SEED from config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def get_config_hash(config_data) -> str:
    """
    Generates a deterministic MD5 hash for a given configuration object.
    This is used to create unique cache keys based on feature configurations
    (e.g., the list of geometric features to extract).

    Args:
        config_data: A JSON-serializable Python object (e.g., list, dict).

    Returns:
        str: The hexadecimal MD5 hash string.
    """
    # Serialize the config data to a JSON string.
    # sort_keys=True ensures that dictionaries with the same content
    # but different insertion orders produce the same string.
    serialized = json.dumps(config_data, sort_keys=True).encode("utf-8")

    # Generate MD5 hash
    return hashlib.md5(serialized).hexdigest()
