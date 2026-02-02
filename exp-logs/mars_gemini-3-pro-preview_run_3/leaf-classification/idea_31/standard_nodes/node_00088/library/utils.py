import os
import random
import pickle
import logging
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(name, log_file, level=logging.INFO):
    """
    Configures and returns a logger that writes to both a file and the console.

    Args:
        name (str): Name of the logger.
        log_file (str): Path to the log file.
        level (int): Logging level (default: logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding multiple handlers if function is called repeatedly
    if not logger.handlers:
        logger.addHandler(handler)
        logger.addHandler(console_handler)

    return logger


def clip_probabilities(probs):
    """
    Clips probabilities to the range [1e-15, 1 - 1e-15] to avoid log loss extremes.

    Args:
        probs (np.ndarray): Probability matrix or vector.

    Returns:
        np.ndarray: Clipped probabilities.
    """
    epsilon = 1e-15
    return np.clip(probs, epsilon, 1.0 - epsilon)


def save_pickle(obj, path):
    """
    Saves an object to a pickle file, ensuring the directory exists.

    Args:
        obj (object): The object to save.
        path (str): The destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    """
    Loads an object from a pickle file.

    Args:
        path (str): The source file path.

    Returns:
        object: The loaded object.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def save_numpy(array, path):
    """
    Saves a numpy array to a .npy file, ensuring the directory exists.

    Args:
        array (np.ndarray): The array to save.
        path (str): The destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, array)


def load_numpy(path):
    """
    Loads a numpy array from a .npy file.

    Args:
        path (str): The source file path.

    Returns:
        np.ndarray: The loaded array.
    """
    return np.load(path, allow_pickle=True)
