import os
import time
import random
import numpy as np
import torch
from library import config


def set_seed(seed: int = config.SEED) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to config.SEED.
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # OS environment (hashing)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Timer:
    """
    Context manager to measure and print the execution time of a code block.

    Usage:
        with Timer("Data Loading"):
            load_data()
    """

    def __init__(self, name: str = "Process"):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        print(f"[{self.name}] Start")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        print(f"[{self.name}] Done in {elapsed_time:.4f} seconds")
