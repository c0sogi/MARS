import os
import random
import warnings
import numpy as np
import torch
from library.config import Config


def set_seed(seed: int = Config.SEED) -> None:
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    # Python random
    random.seed(seed)

    # Environment variable for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Ensure deterministic behavior in CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Returns the PyTorch device to be used for computation.

    Returns:
        torch.device: The device (cuda or cpu) defined in Config.
    """
    return torch.device(Config.DEVICE)


def suppress_warnings() -> None:
    """
    Suppresses unnecessary warnings to keep the output clean.
    """
    warnings.filterwarnings("ignore")
    # Suppress specific library warnings if necessary (e.g., Transformers)
    os.environ["transformers_verbosity"] = "error"
