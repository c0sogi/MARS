import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Returns the appropriate torch device (CUDA if available, else CPU).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def log_message(message: str, log_file: str = None):
    """
    Prints a message to stdout and optionally appends it to a log file.
    """
    print(message)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "a") as f:
            f.write(message + "\n")


def print_metrics(metrics: dict, prefix: str = ""):
    """
    Prints validation metrics with full precision without rounding.
    """
    items = []
    for k, v in metrics.items():
        items.append(f"{k}: {v}")

    msg = " ".join(items)
    if prefix:
        msg = f"[{prefix}] {msg}"
    print(msg)


def save_numpy_cache(data: np.ndarray, filename: str):
    """
    Saves a numpy array to the cache directory defined in Config.
    Strictly uses .npy format.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    file_path = os.path.join(Config.CACHE_DIR, f"{filename}.npy")
    np.save(file_path, data)


def load_numpy_cache(filename: str):
    """
    Loads a numpy array from the cache directory.
    Returns None if the file does not exist.
    """
    file_path = os.path.join(Config.CACHE_DIR, f"{filename}.npy")
    if os.path.exists(file_path):
        return np.load(file_path)
    return None
