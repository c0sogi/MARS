import os
import time
import random
import warnings
import numpy as np
import torch
from library.config import Config


def suppress_warnings():
    """
    Suppress warnings from various libraries to keep stdout clean.
    """
    warnings.filterwarnings("ignore")
    # Suppress TensorFlow logs if imported implicitly by other libraries
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def set_seed(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # OS environment for hashing
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic algorithms for PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Timer:
    """
    Context manager to measure and print the execution time of a code block.
    """

    def __init__(self, name: str = "Task"):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        print(f"[{self.name}] Starting...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        print(f"[{self.name}] Completed in {elapsed_time:.6f} seconds.")
