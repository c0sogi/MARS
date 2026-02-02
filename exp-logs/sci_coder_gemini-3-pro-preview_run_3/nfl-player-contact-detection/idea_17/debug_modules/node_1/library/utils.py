import os
import random
import numpy as np
import hashlib
import json
import torch
from sklearn.metrics import matthews_corrcoef
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # PyTorch seeding
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calc_mcc(y_true, y_pred):
    """
    Calculates the Matthews Correlation Coefficient.

    Args:
        y_true (np.array): Ground truth binary labels.
        y_pred (np.array): Predicted binary labels.

    Returns:
        float: The MCC score.
    """
    return matthews_corrcoef(y_true, y_pred)


def optimize_thresholds(y_true, y_pred_proba, num_steps=100):
    """
    Performs a linear search to find the probability threshold that maximizes MCC.

    Args:
        y_true (np.array): Ground truth binary labels.
        y_pred_proba (np.array): Predicted probabilities.
        num_steps (int): Number of steps in the linear search between 0 and 1.

    Returns:
        tuple: (best_threshold, best_mcc)
    """
    best_mcc = -1.0
    best_thresh = 0.5

    # Generate thresholds excluding 0 and 1 to avoid edge cases if desired,
    # but 0-1 inclusive is standard for search.
    thresholds = np.linspace(0.01, 0.99, num_steps)

    for thresh in thresholds:
        # Binarize predictions
        y_pred = (y_pred_proba >= thresh).astype(int)

        # Calculate metric
        mcc = matthews_corrcoef(y_true, y_pred)

        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = thresh

    return best_thresh, best_mcc


def get_hashed_cache_path(base_name, config_dict, extension=".parquet"):
    """
    Generates a cache file path including a hash of the configuration dictionary.

    Args:
        base_name (str): The base name for the file (e.g., 'streamA_train').
        config_dict (dict): Dictionary of configuration parameters affecting the data.
        extension (str): File extension (default: .parquet).

    Returns:
        str: Full path to the cached file in the working directory.
    """
    # Serialize config to a JSON string with sorted keys for consistency
    # default=str handles non-serializable types by converting them to string
    config_str = json.dumps(config_dict, sort_keys=True, default=str)

    # Generate MD5 hash of the config string
    config_hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()

    # Construct the filename
    filename = f"{base_name}_{config_hash}{extension}"

    # Return full path
    return os.path.join(Config.WORKING_DIR, filename)
