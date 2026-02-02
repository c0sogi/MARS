import os
import random
import numpy as np
import hashlib
import json
from library.config import (
    SEED,
    EFD_HARMONICS,
    SPATIAL_FEATURES,
    BINARY_THRESHOLD,
    NUM_MARGIN_FEATURES,
    NUM_SHAPE_FEATURES,
    NUM_TEXTURE_FEATURES,
    FLOAT_PRECISION,
)


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python's random module,
    numpy, and environment variables.

    Args:
        seed (int): The seed value to use. Defaults to the global SEED constant.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # Attempt to set torch seed if available, as it's a common dependency
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_config_hash():
    """
    Generates a unique MD5 hash based on the current feature extraction configuration.
    This hash is used to version cached datasets, ensuring that changes to
    hyperparameters (like EFD harmonics or spatial features) invalidate old caches.

    Returns:
        str: A hexadecimal string representing the configuration hash.
    """
    config_dict = {
        "EFD_HARMONICS": EFD_HARMONICS,
        "SPATIAL_FEATURES": sorted(SPATIAL_FEATURES),
        "BINARY_THRESHOLD": BINARY_THRESHOLD,
        "NUM_MARGIN_FEATURES": NUM_MARGIN_FEATURES,
        "NUM_SHAPE_FEATURES": NUM_SHAPE_FEATURES,
        "NUM_TEXTURE_FEATURES": NUM_TEXTURE_FEATURES,
        "FLOAT_PRECISION": str(FLOAT_PRECISION),
    }

    # Serialize to JSON with sorted keys to ensure determinism
    config_str = json.dumps(config_dict, sort_keys=True)

    # Compute MD5 hash
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()
