import os
import random
import numpy as np
import torch
import ast
from library.config import Config


def set_seed(seed=Config.RANDOM_STATE):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.RANDOM_STATE.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_stringified_list(list_str):
    """
    Parses a string representation of a list (e.g., "['a', 'b']") back into a Python list.
    This is used to recover list objects from CSV columns like 'requester_subreddits_at_request'.

    Args:
        list_str (str): The string representation of the list.

    Returns:
        list: The parsed Python list. Returns an empty list if parsing fails or input is not a string.
    """
    if pd_isna(list_str):
        return []

    if not isinstance(list_str, str):
        # If it's already a list, return it
        if isinstance(list_str, list):
            return list_str
        return []

    try:
        # ast.literal_eval is safe for evaluating strings containing Python literals
        parsed = ast.literal_eval(list_str)
        if isinstance(parsed, list):
            return parsed
        return []
    except (ValueError, SyntaxError):
        # Return empty list on failure
        return []


def pd_isna(obj):
    """
    Helper to check for NA/NaN values without importing pandas directly if not needed,
    or handling standard numpy nan types.
    """
    if obj is None:
        return True
    try:
        if np.isnan(obj):
            return True
    except (TypeError, ValueError):
        pass
    return False
