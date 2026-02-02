import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducible results.

    Args:
        seed (int): The seed value to use. Defaults to the value in Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # PyTorch seeding
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clip_probabilities(probs: np.ndarray) -> np.ndarray:
    """
    Clips predicted probabilities to the range [1e-15, 1-1e-15] to avoid
    mathematical errors (e.g., log(0)) during metric calculation.

    Args:
        probs (np.ndarray): The array of predicted probabilities.

    Returns:
        np.ndarray: The array with values clipped within the safe range.
    """
    return np.clip(probs, Config.PROB_CLIP_MIN, Config.PROB_CLIP_MAX)
