import os
import random
import json
import hashlib
import numpy as np
import torch
from dataclasses import asdict, is_dataclass
from sklearn.metrics import matthews_corrcoef
from library.config import CACHE_DIR, SEED


def set_seed(seed: int = SEED) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed for consistent hashing of strings/objects in some contexts
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Matthews Correlation Coefficient (MCC) between ground truth and predictions.

    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_pred (np.ndarray): Predicted binary labels.

    Returns:
        float: The MCC score.
    """
    # Ensure inputs are numpy arrays for safety
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return matthews_corrcoef(y_true, y_pred)


def get_hashed_filepath(
    base_filename: str, config_obj: any, extension: str = "parquet"
) -> str:
    """
    Generates a file path with a hash suffix derived from a configuration object.
    This enables parameter-aware caching by ensuring unique filenames for different configurations.

    Args:
        base_filename (str): The prefix for the filename (e.g., 'train_features').
        config_obj (any): The configuration object (dataclass, dict, list, etc.) to hash.
        extension (str): The file extension (default: 'parquet').

    Returns:
        str: Full path to the file in the CACHE_DIR.
    """
    # Convert dataclass to dict if necessary
    if is_dataclass(config_obj):
        config_data = asdict(config_obj)
    else:
        config_data = config_obj

    # Serialize to JSON string for consistent hashing.
    # sort_keys=True ensures that dictionary key order doesn't affect the hash.
    # default=str handles types that aren't natively JSON serializable (e.g., functions/lambdas).
    try:
        config_str = json.dumps(config_data, sort_keys=True, default=str)
    except TypeError:
        # Fallback if json serialization fails completely
        config_str = str(config_data)

    # Compute MD5 hash of the configuration string
    md5_hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()

    # Construct the final filename
    filename = f"{base_filename}_{md5_hash}.{extension}"

    # Return the full path
    return os.path.join(CACHE_DIR, filename)
