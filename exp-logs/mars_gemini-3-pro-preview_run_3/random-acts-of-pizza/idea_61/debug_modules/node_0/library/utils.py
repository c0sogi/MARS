import os
import time
import random
import joblib
import numpy as np
import torch
from contextlib import contextmanager


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # Torch might not be installed or CUDA not available; fail silently
        pass


@contextmanager
def Timer(name):
    """
    Context manager to measure and print the execution time of a code block.

    Args:
        name (str): The name of the operation being timed.
    """
    t0 = time.time()
    yield
    t1 = time.time()
    print(f"[{name}] done in {t1 - t0:.5f} s")


def save_joblib(obj, path):
    """
    Saves a Python object to disk using joblib (pickling).
    Ensures the target directory exists before saving.

    Args:
        obj: The Python object to save.
        path (str): The file path to save to.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path):
    """
    Loads a Python object from disk using joblib.

    Args:
        path (str): The file path to load from.

    Returns:
        The loaded Python object.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return joblib.load(path)
