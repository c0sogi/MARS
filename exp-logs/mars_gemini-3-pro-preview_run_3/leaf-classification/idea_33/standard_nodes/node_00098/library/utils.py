import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Determines and returns the appropriate PyTorch device.

    Returns:
        torch.device: 'cuda' if a GPU is available, otherwise 'cpu'.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def clip_probabilities(data):
    """
    Clips probability values to the range [1e-15, 1 - 1e-15] to avoid
    extremes in the log loss metric.

    Args:
        data (np.ndarray or pd.DataFrame): The probability data to clip.

    Returns:
        The data with values clipped to the range defined in Config.
    """
    min_val = Config.PROB_CLIP_MIN
    max_val = Config.PROB_CLIP_MAX

    if isinstance(data, pd.DataFrame):
        return data.clip(lower=min_val, upper=max_val)
    elif isinstance(data, np.ndarray):
        return np.clip(data, min_val, max_val)
    else:
        # Fallback for lists or other iterables, converting to numpy array
        return np.clip(np.array(data), min_val, max_val)
