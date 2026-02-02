import os
import random
import time
import numpy as np
import torch
from contextlib import contextmanager


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # PyTorch seeding
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@contextmanager
def timer(name: str):
    """
    Context manager to measure and print the execution time of a code block.

    Args:
        name (str): The name of the operation being timed.
    """
    t0 = time.time()
    print(f"[{name}] Starting...")
    yield
    elapsed = time.time() - t0
    print(f"[{name}] Done in {elapsed:.3f} s")


def print_header(title: str):
    """
    Prints a formatted header to the console to separate pipeline stages.

    Args:
        title (str): The title to display.
    """
    print(f"\n{'='*10} {title} {'='*10}")
