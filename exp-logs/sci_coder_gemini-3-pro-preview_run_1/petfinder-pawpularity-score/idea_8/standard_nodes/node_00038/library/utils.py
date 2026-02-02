import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Returns the appropriate PyTorch device (CUDA if available, else CPU).

    Returns:
        torch.device: The device to be used for computation.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(file_path: str):
    """
    Ensures that the directory for the given file path exists.

    Args:
        file_path (str): The path to the file.
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def save_array(file_path: str, array: np.ndarray):
    """
    Saves a NumPy array to a file using .npy format, creating directories if necessary.

    Args:
        file_path (str): The path where the array should be saved.
        array (np.ndarray): The array to save.
    """
    ensure_dir(file_path)
    np.save(file_path, array)


def load_array(file_path: str) -> np.ndarray:
    """
    Loads a NumPy array from a .npy file.

    Args:
        file_path (str): The path to the .npy file.

    Returns:
        np.ndarray: The loaded array.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return np.load(file_path)


def print_metric(name: str, value: float):
    """
    Prints a metric name and its value with full precision.

    Args:
        name (str): The name of the metric.
        value (float): The value of the metric.
    """
    print(f"{name}: {value}")
