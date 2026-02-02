import torch
import os
import random
import numpy as np

# Import the provided seed_everything function to avoid re-implementation
# and maintain consistency with the project configuration.
from library.config import seed_everything


def get_device(force_cpu: bool = False) -> torch.device:
    """
    Determines the computing device to use.

    Args:
        force_cpu (bool): If True, forces the usage of CPU even if GPU is available.
                          Defaults to False.

    Returns:
        torch.device: The torch device object ('cuda' or 'cpu').
    """
    if not force_cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
