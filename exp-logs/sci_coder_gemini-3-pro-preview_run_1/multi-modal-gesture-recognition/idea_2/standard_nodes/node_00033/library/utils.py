import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    """
    Returns the appropriate PyTorch device (CUDA or CPU).

    Returns:
        torch.device: The device object.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def ensure_dir(path):
    """
    Ensures that the directory for a given path exists.

    Args:
        path (str): The directory path or file path.
    """
    if os.path.splitext(path)[1]:  # It's a file
        dirname = os.path.dirname(path)
    else:  # It's a directory
        dirname = path

    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)


def save_npz(path, data_dict):
    """
    Saves a dictionary of numpy arrays to a compressed .npz file.
    This avoids using pickle as per requirements.

    Args:
        path (str): The output file path (e.g., 'data.npz').
        data_dict (dict): Dictionary where keys are names and values are numpy arrays.
    """
    ensure_dir(path)
    np.savez_compressed(path, **data_dict)


def load_npz(path):
    """
    Loads a compressed .npz file and returns it as a dictionary.

    Args:
        path (str): The path to the .npz file.

    Returns:
        dict: A dictionary containing the loaded arrays.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache file not found: {path}")

    try:
        with np.load(path) as data:
            # Convert NpzFile object to a standard dictionary to keep data in memory
            # after the context manager closes the file.
            return {key: data[key] for key in data.files}
    except Exception as e:
        print(f"Error loading cache from {path}: {e}")
        return None
