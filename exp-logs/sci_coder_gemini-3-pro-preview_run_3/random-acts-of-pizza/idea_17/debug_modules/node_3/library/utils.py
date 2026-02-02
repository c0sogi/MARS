import os
import time
import random
import numpy as np
import torch
import functools
from contextlib import contextmanager


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python's random, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    # Python random
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # OS environment
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def time_execution(func):
    """
    Decorator to measure and print the execution time of a function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Function '{func.__name__}' executed in {elapsed:.4f} seconds.")
        return result

    return wrapper


@contextmanager
def timer(description: str):
    """
    Context manager to measure and print the execution time of a code block.

    Args:
        description (str): Label for the code block being timed.
    """
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"[{description}] completed in {elapsed:.4f} seconds.")


def print_metric(name: str, value: float):
    """
    Prints a metric name and its value with full precision.

    Args:
        name (str): The name of the metric.
        value (float): The value of the metric.
    """
    print(f"{name}: {value}")
