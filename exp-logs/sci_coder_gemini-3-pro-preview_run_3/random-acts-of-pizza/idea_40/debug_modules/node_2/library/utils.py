import os
import sys
import time
import random
import warnings
import contextlib
import numpy as np
import torch
from library.config import Config


def set_seed(seed: int = Config.RANDOM_SEED) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.RANDOM_SEED.
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # OS environment
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # Pass if torch is not installed or other issues arise
        pass


def suppress_warnings() -> None:
    """
    Suppresses warnings and verbose logs from libraries to ensure clean output.
    """
    warnings.filterwarnings("ignore")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["LGBM_VERBOSITY"] = "-1"


class Timer(contextlib.ContextDecorator):
    """
    Context manager to measure and print the execution time of a block of code.
    """

    def __init__(self, name: str = "Task"):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        print(f"[{self.name}] Starting...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        print(f"[{self.name}] Completed in {elapsed:.6f} seconds.")
        return False  # Propagate exceptions if any


def print_metric(name: str, value: float) -> None:
    """
    Prints a metric name and its value with full precision.

    Args:
        name (str): The name of the metric.
        value (float): The value of the metric.
    """
    print(f"{name}: {value}")
