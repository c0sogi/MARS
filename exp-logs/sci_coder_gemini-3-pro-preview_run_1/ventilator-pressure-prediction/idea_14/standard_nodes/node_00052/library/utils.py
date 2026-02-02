import os
import random
import hashlib
import json
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CUDNN backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_config_hash(config_obj) -> str:
    """
    Generates a unique MD5 hash for a given configuration object.
    This is used to create unique cache filenames based on feature lists or
    hyperparameter configurations.

    Args:
        config_obj: A list (e.g., feature names) or dictionary (config) to hash.

    Returns:
        str: The hexadecimal MD5 hash string.
    """
    if isinstance(config_obj, dict):
        # Sort keys to ensure the hash is deterministic regardless of key insertion order
        encoded = json.dumps(config_obj, sort_keys=True).encode("utf-8")
    elif isinstance(config_obj, list):
        # For lists (like features), we preserve order as it might matter for channel mapping,
        # or we could sort if order invariance is desired. Given the pipeline context,
        # usually feature order is fixed in the config.
        encoded = json.dumps(config_obj).encode("utf-8")
    else:
        # Fallback for other serializable types
        encoded = str(config_obj).encode("utf-8")

    return hashlib.md5(encoded).hexdigest()


def compute_metric(y_pred, y_true, u_out):
    """
    Computes the Mean Absolute Error (MAE) between predicted and actual pressures,
    scored ONLY during the inspiratory phase (u_out == 0).

    Args:
        y_pred: Predicted pressure values (numpy array or torch tensor).
        y_true: Actual pressure values (numpy array or torch tensor).
        u_out: Control input u_out (numpy array or torch tensor), where 0 indicates inspiration.

    Returns:
        float: The MAE calculated over the inspiratory phase.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(u_out, torch.Tensor):
        u_out = u_out.detach().cpu().numpy()

    # Flatten arrays to handle shapes like (Batch, Time) vs (Batch*Time,)
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    u_out = u_out.flatten()

    # Create a boolean mask for the inspiratory phase (u_out == 0)
    # u_out is binary: 0 for inspiration, 1 for expiration.
    mask = u_out == 0

    # If there are no inspiratory steps (unlikely in valid data), return 0.0
    if np.sum(mask) == 0:
        return 0.0

    # Compute MAE only on the masked elements
    mae = np.mean(np.abs(y_pred[mask] - y_true[mask]))

    return float(mae)
