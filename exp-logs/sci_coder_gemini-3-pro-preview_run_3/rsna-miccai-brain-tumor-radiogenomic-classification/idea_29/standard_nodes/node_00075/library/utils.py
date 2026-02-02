import torch
import os
import numpy as np
import random
from library.config import Config, seed_everything


def get_device():
    """
    Returns the PyTorch device (GPU or CPU) configured for the system.

    Returns:
        torch.device: The device to be used for computation.
    """
    return Config.DEVICE
