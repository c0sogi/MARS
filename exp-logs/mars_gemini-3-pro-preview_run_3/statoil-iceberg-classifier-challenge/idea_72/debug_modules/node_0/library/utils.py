import os
import random
import numpy as np
import torch
from library import config


def set_seed(seed=config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to the value in config.SEED.
    """
    # Python random
    random.seed(seed)

    # Environment variable for hash seeding
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

        # Ensure deterministic behavior in cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Random seed set to: {seed}")


def get_device():
    """
    Checks for GPU availability and returns the appropriate PyTorch device.

    Returns:
        torch.device: The device object (cuda or cpu).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device selected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Device selected: CPU")

    return device
