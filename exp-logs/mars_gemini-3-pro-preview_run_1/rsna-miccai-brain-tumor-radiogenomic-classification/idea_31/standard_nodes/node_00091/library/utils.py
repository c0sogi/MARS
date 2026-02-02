import torch
import os
import random
import numpy as np

# Import existing utility functions from the provided configuration file
# to avoid re-implementation and ensure consistency.
from library.config import seed_everything, load_dicom_image


def get_device() -> torch.device:
    """
    Selects the GPU if available, otherwise falls back to CPU.

    Returns:
        torch.device: The device object (cuda or cpu).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
