import os
import random
import numpy as np
import json
import hashlib
import torch
import library.config as config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Ensure deterministic behavior in cudnn
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        # Torch might not be installed or used, which is acceptable
        pass


def generate_config_hash():
    """
    Generates a unique MD5 hash based on the active feature configuration and
    preprocessing parameters defined in library.config.

    This hash is used to validate cached datasets. If the feature columns,
    preprocessing method, or precision settings change, the hash will change,
    triggering a re-computation of the dataset.

    Returns:
        str: A hexadecimal MD5 hash string representing the current configuration.
    """
    # Collect configuration parameters that affect data content and structure.
    # We exclude model hyperparameters (like OAS_PARAMS) because changing them
    # should not require re-extracting features from images.
    config_state = {
        "image_feature_cols": sorted(config.IMAGE_FEATURE_COLS),
        "tabular_feature_cols": sorted(config.TABULAR_FEATURE_COLS),
        "preprocess_method": config.PREPROCESS_POWER_METHOD,
        "float_precision": str(config.FLOAT_PRECISION),
        "cache_version": config.CACHE_VERSION,
    }

    # Serialize to JSON. sort_keys=True is critical for deterministic hashing.
    config_str = json.dumps(config_state, sort_keys=True)

    # Compute MD5 hash
    hasher = hashlib.md5()
    hasher.update(config_str.encode("utf-8"))

    return hasher.hexdigest()
