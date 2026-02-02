import os
import random
import numpy as np
import torch
import hashlib
import json
from library import config


def set_seed(seed: int = config.SEED) -> None:
    """
    Sets the random seed for Python's random module, NumPy, and PyTorch
    to ensure reproducibility across the entire pipeline.

    Args:
        seed (int): The seed value to use. Defaults to config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Ensure deterministic behavior in CuDNN
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        # If torch is not installed or fails, we proceed without it
        pass


def get_config_hash() -> str:
    """
    Generates a unique MD5 hash based on the current configuration parameters.
    This hash is used to version cached datasets. If any critical hyperparameter
    in config.py changes, this hash will change, forcing the pipeline to
    re-compute features rather than loading stale cache.

    Returns:
        str: A hexadecimal MD5 hash string representing the current config state.
    """
    # dictionary of critical parameters that affect data processing and feature extraction
    config_state = {
        "SEED": config.SEED,
        "BINARY_THRESHOLD_VALUE": config.BINARY_THRESHOLD_VALUE,
        "BINARY_THRESHOLD_TYPE": config.BINARY_THRESHOLD_TYPE,
        "CONTOUR_MODE": config.CONTOUR_MODE,
        "GEOMETRIC_FEATURES": config.GEOMETRIC_FEATURES,
        "VARIANCE_THRESHOLD": config.VARIANCE_THRESHOLD,
        # Convert type to string for serialization
        "FLOAT_PRECISION": str(config.FLOAT_PRECISION),
        "EXCLUDE_COLUMNS": config.EXCLUDE_COLUMNS,
        "PROB_CLIP_EPS": config.PROB_CLIP_EPS,
    }

    # Serialize to JSON with sorted keys to ensure deterministic ordering
    config_str = json.dumps(config_state, sort_keys=True)

    # Generate MD5 hash
    config_hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()

    return config_hash
