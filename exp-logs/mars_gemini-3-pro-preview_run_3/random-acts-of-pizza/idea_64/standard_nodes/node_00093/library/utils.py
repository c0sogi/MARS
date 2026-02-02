import os
import time
import random
import numpy as np
import torch
import contextlib
from library.config import Config


def set_seed(seed: int = Config.SEED):
    """
    Sets the seed for random number generators in Python, NumPy, and PyTorch
    to ensure reproducible results.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
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


@contextlib.contextmanager
def Timer(name: str):
    """
    Context manager to measure and print the execution time of a block of code.

    Args:
        name (str): The name of the operation being timed.
    """
    t0 = time.time()
    print(f"[{name}] Start")
    try:
        yield
    finally:
        t1 = time.time()
        elapsed = t1 - t0
        print(f"[{name}] Done in {elapsed} seconds")
