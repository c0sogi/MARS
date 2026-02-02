import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    This function ensures that the random number generation is deterministic,
    which is crucial for reproducing the specific initialization and data
    shuffling dynamics described in the solution strategy.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    # Python random
    random.seed(seed)

    # Environment variable for hash seeding
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy random
    np.random.seed(seed)

    # PyTorch random
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Force deterministic algorithms for cuDNN
    # This may impact performance but is necessary for exact reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Returns the appropriate PyTorch device (GPU if available, else CPU).

    Returns:
        torch.device: The device object ('cuda' or 'cpu').
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
