import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU.

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log1p_target(y):
    """
    Applies log1p transformation to the target variable.
    Used to compress the dynamic range of time_to_eruption for better model convergence.

    Args:
        y (np.array or float): The target value(s).

    Returns:
        np.array or float: log(y + 1)
    """
    return np.log1p(y)


def expm1_target(y):
    """
    Applies expm1 transformation to inverse the log-scaling.
    Used to convert model predictions back to the original time scale.

    Args:
        y (np.array or float): The log-scaled prediction(s).

    Returns:
        np.array or float: exp(y) - 1
    """
    return np.expm1(y)
