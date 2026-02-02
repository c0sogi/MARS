import os
import time
import random
import pickle
import numpy as np
import torch
from library.config import Config


def set_seed(seed: int = Config.seed):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Configures CuDNN for deterministic execution.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_pickle(obj, path: str):
    """
    Saves a Python object to a file using pickle.
    Ensures the parent directory exists before saving.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    """
    Loads a Python object from a pickle file.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


class Timer:
    """
    Context manager to measure and print the execution time of a code block.
    """

    def __init__(self, name: str = "Process"):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        print(f"[{self.name}] completed in {elapsed_time:.6f} seconds")
