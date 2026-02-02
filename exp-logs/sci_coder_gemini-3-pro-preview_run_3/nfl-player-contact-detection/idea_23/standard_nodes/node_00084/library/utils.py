import os
import random
import numpy as np
import hashlib
import json
from sklearn.metrics import matthews_corrcoef

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_mcc(y_true, y_pred):
    """
    Calculates the Matthews Correlation Coefficient (MCC).

    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets as returned by a classifier.

    Returns:
        float: The MCC score.
    """
    # Ensure inputs are numpy arrays for consistency
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return matthews_corrcoef(y_true, y_pred)


def generate_hash(config_dict: dict) -> str:
    """
    Generates a unique MD5 hash based on a configuration dictionary.
    This is used to create unique identifiers for caching intermediate files.

    Args:
        config_dict (dict): Dictionary containing configuration parameters.
                            Can include nested dictionaries or lists.

    Returns:
        str: A hexadecimal MD5 hash string.
    """

    # Helper to handle non-serializable objects (like sets or numpy types)
    def default_serializer(obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return sorted(list(obj))
        return str(obj)

    # Sort keys to ensure deterministic JSON string
    encoded = json.dumps(
        config_dict, sort_keys=True, default=default_serializer
    ).encode("utf-8")
    return hashlib.md5(encoded).hexdigest()
