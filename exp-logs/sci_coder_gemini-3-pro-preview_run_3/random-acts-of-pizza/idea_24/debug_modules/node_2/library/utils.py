import os
import random
import time
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set. Defaults to 42.
    """
    # Python's built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # OS Hash Seed (for dictionary ordering/hashing consistency)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in PyTorch backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Timer:
    """
    Context manager to measure and print the execution time of a code block.
    """

    def __init__(self, name="Task"):
        """
        Initialize the Timer.

        Args:
            name (str): The name of the task being timed.
        """
        self.name = name
        self.start = None
        self.end = None

    def __enter__(self):
        self.start = time.time()
        print(f"[{self.name}] Starting...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        elapsed = self.end - self.start
        # Print with high precision as requested
        print(f"[{self.name}] Completed in {elapsed:.6f} seconds")


def print_header(title):
    """
    Prints a formatted header to separate log sections.

    Args:
        title (str): The title to display in the header.
    """
    print(f"\n{'='*10} {title} {'='*10}")
