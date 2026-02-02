import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    This function configures:
    - Python's built-in random module
    - NumPy's random number generator
    - PyTorch's CPU and CUDA random number generators
    - Python hash seed environment variable
    - PyTorch CuDNN backends for deterministic execution

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Python environment
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in CuDNN
    # Note: This may impact performance but is necessary for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Automatically selects the available hardware device.

    Prioritizes CUDA (GPU) if available, otherwise falls back to CPU.

    Returns:
        torch.device: The selected device object.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
