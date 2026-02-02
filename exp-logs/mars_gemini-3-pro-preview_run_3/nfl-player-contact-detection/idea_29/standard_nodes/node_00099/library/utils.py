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
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_config_hash(config_dict: dict) -> str:
    """
    Generates a unique MD5 hash based on a configuration dictionary.
    Useful for creating cache keys that invalidate when configuration changes.

    Args:
        config_dict (dict): The configuration dictionary to hash.

    Returns:
        str: The hexadecimal MD5 hash string.
    """

    # specific handler to convert sets to lists for json serialization
    # and other non-serializable types if necessary
    def default_converter(o):
        if isinstance(o, set):
            return sorted(list(o))
        if isinstance(
            o,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(o)
        if isinstance(o, (np.float_, np.float16, np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        return str(o)

    # Sort keys to ensure consistent ordering for hashing
    encoded_config = json.dumps(
        config_dict, sort_keys=True, default=default_converter
    ).encode("utf-8")

    return hashlib.md5(encoded_config).hexdigest()
