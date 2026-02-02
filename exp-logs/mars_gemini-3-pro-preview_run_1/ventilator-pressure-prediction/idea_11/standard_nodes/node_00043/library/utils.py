import os
import random
import numpy as np
import torch
import hashlib
import json


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_hash_filename(config_dict: dict, prefix: str) -> str:
    """
    Generates a unique filename based on the MD5 hash of a configuration dictionary.
    Used for caching processed datasets to ensure data consistency with configuration.

    Args:
        config_dict (dict): Dictionary containing configuration parameters
                            (e.g., feature lists, sequence length).
        prefix (str): Prefix for the filename (e.g., 'train_x', 'scaler').

    Returns:
        str: A filename string in the format '{prefix}_{hash}.npy'.
    """
    # Create a copy to avoid modifying the original dict
    conf = config_dict.copy()

    # Sort keys to ensure consistent ordering for hashing
    # Convert to string representation to handle non-JSON serializable objects if necessary
    # using sort_keys=True in json.dumps is robust for standard types
    try:
        dict_str = json.dumps(conf, sort_keys=True)
    except TypeError:
        # Fallback for non-serializable objects: stringify the sorted items
        dict_str = str(sorted(conf.items()))

    # Compute MD5 hash
    md5_hash = hashlib.md5(dict_str.encode("utf-8")).hexdigest()

    return f"{prefix}_{md5_hash}.npy"


def masked_mae_metric(
    y_pred: torch.Tensor, y_true: torch.Tensor, u_out: torch.Tensor
) -> float:
    """
    Calculates the Mean Absolute Error (MAE) masked by the inspiratory phase.
    The metric is only computed where u_out == 0.

    Args:
        y_pred (torch.Tensor): Predicted pressure values.
        y_true (torch.Tensor): Actual pressure values.
        u_out (torch.Tensor): Control input indicating expiratory phase (1) or inspiratory (0).

    Returns:
        float: The masked MAE value.
    """
    # Ensure inputs are on the same device
    if y_pred.device != y_true.device:
        y_true = y_true.to(y_pred.device)
    if u_out.device != y_pred.device:
        u_out = u_out.to(y_pred.device)

    # Create mask: we want to score where u_out is 0 (inspiratory phase)
    # u_out is 0 or 1. mask = 1 - u_out gives 1 for inspiratory, 0 for expiratory.
    mask = 1 - u_out

    # Calculate absolute error
    absolute_error = torch.abs(y_pred - y_true)

    # Apply mask
    masked_error = absolute_error * mask

    # Calculate mean: Sum of errors / Sum of mask elements (count of inspiratory steps)
    # Add a small epsilon to avoid division by zero (though unlikely in this dataset)
    score = masked_error.sum() / (mask.sum() + 1e-8)

    return score.item()
