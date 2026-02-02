import os
import random
import numpy as np
import torch
import json
import hashlib
from sklearn.metrics import matthews_corrcoef
from library.config import (
    STREAM_A_BASE_FEATURES,
    STREAM_B_BASE_FEATURES,
    WINDOW_SIZE,
    LAG_STEPS,
    SAMPLING_RATIO,
)


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across various libraries.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calc_mcc(y_true, y_pred):
    """
    Calculates the Matthews Correlation Coefficient.

    Args:
        y_true: Array-like of ground truth labels.
        y_pred: Array-like of predicted labels.

    Returns:
        float: The MCC score.
    """
    return matthews_corrcoef(y_true, y_pred)


def get_config_hash():
    """
    Generates a unique hash based on the current feature configuration and
    preprocessing constants. This is used for cache invalidation.

    Returns:
        str: MD5 hash hexdigest of the configuration.
    """
    config_dict = {
        "STREAM_A_BASE_FEATURES": sorted(STREAM_A_BASE_FEATURES),
        "STREAM_B_BASE_FEATURES": sorted(STREAM_B_BASE_FEATURES),
        "WINDOW_SIZE": WINDOW_SIZE,
        "LAG_STEPS": LAG_STEPS,
        "SAMPLING_RATIO": SAMPLING_RATIO,
    }

    # Serialize to JSON with sorted keys to ensure consistency
    config_str = json.dumps(config_dict, sort_keys=True)

    # Generate MD5 hash
    config_hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()

    return config_hash
