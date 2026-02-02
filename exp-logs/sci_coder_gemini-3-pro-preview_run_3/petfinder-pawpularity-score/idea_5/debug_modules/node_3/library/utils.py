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

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_config_hash(config) -> str:
    """
    Generates an MD5 hash based on the configuration parameters that affect
    feature extraction and preprocessing. This is used to create unique
    filenames for cached features.

    Args:
        config: The Config class or object containing settings.

    Returns:
        str: A hexadecimal MD5 hash string.
    """
    # Select attributes that affect feature extraction and data processing.
    # We intentionally exclude training hyperparameters (like SVR_C, LGBM_PARAMS)
    # so that tuning them doesn't trigger expensive re-extraction of features.
    config_dict = {
        "BACKBONES": config.BACKBONES,
        "IMAGE_SIZE": config.IMAGE_SIZE,
        "PCA_VARIANCE": config.PCA_VARIANCE,
        "SEED": config.SEED,
    }

    # Serialize to JSON with sorted keys to ensure consistent string representation
    config_str = json.dumps(config_dict, sort_keys=True)

    # Generate MD5 hash
    md5_hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()

    return md5_hash
