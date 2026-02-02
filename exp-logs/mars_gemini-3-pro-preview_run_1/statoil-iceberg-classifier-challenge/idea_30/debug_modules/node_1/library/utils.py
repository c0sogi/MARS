import os
import random
import numpy as np
import torch

# Global statistical constants derived from the training set analysis (Data Analysis Step).
# These are used for the 'Independent Band Normalization' strategy.
# Band 1 (HH)
BAND1_MIN = -45.5944
BAND1_MAX = 32.1806
BAND1_MEAN = -20.5754
BAND1_STD = 5.2486

# Band 2 (HV)
BAND2_MIN = -45.6555
BAND2_MAX = 17.8628
BAND2_MEAN = -26.2593
BAND2_STD = 3.3965


def set_seed(seed=42):
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
    # Deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def min_max_scale(data, min_val, max_val):
    """
    Applies global Min-Max scaling to the input data using provided bounds.
    Formula: (data - min) / (max - min)

    Args:
        data (np.ndarray or torch.Tensor): Input data to be normalized.
        min_val (float): The global minimum value for this band.
        max_val (float): The global maximum value for this band.

    Returns:
        The normalized data, scaled approximately to [0, 1].
    """
    denom = max_val - min_val
    if denom == 0:
        return data * 0.0
    return (data - min_val) / denom
