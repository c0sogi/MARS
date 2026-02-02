import os
import random
import numpy as np
import torch
import hashlib
import json
from library.config import Config


def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Returns the appropriate PyTorch device (CUDA or CPU).

    Returns:
        torch.device: The device object.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_data_hash(config_cls=Config) -> str:
    """
    Generates a unique MD5 hash based on the configuration parameters that affect
    data processing (Features, Sequence Length, Debug mode).

    This hash is used to version cached data files.

    Args:
        config_cls: The configuration class containing parameters.

    Returns:
        str: A hexadecimal hash string.
    """
    # Dictionary of parameters that determine the data structure/content
    params = {
        "features": config_cls.FEATURES,
        "seq_len": config_cls.SEQ_LEN,
        "debug": config_cls.DEBUG,
        # Including seed ensures different splits/shuffles generate different caches if needed,
        # though usually splits are fixed by metadata.
        "seed": config_cls.SEED,
    }

    # Serialize to JSON with sorting to ensure deterministic string representation
    params_str = json.dumps(params, sort_keys=True)

    # Compute MD5 hash
    return hashlib.md5(params_str.encode("utf-8")).hexdigest()


def save_npy(data: np.ndarray, path: str) -> None:
    """
    Saves a numpy array to a file, ensuring the directory exists.

    Args:
        data (np.ndarray): The data to save.
        path (str): The file path (should end in .npy).
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    np.save(path, data)


def load_npy(path: str) -> np.ndarray:
    """
    Loads a numpy array from a file.

    Args:
        path (str): The file path.

    Returns:
        np.ndarray: The loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache file not found at {path}")
    return np.load(path)


def compute_metric(preds: np.ndarray, targets: np.ndarray, u_out: np.ndarray) -> float:
    """
    Calculates the Mean Absolute Error (MAE) for the inspiratory phase.
    The expiratory phase (where u_out == 1) is excluded from the metric.

    Args:
        preds (np.ndarray): Predicted pressure values.
        targets (np.ndarray): Actual pressure values.
        u_out (np.ndarray): Control input indicating expiratory phase (1) or inspiratory (0).

    Returns:
        float: The MAE for the inspiratory phase.
    """
    # Ensure inputs are numpy arrays
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(u_out, torch.Tensor):
        u_out = u_out.detach().cpu().numpy()

    # Flatten arrays if they are not 1D (e.g., coming from batches)
    preds = preds.flatten()
    targets = targets.flatten()
    u_out = u_out.flatten()

    # Create mask for inspiratory phase (u_out == 0)
    # Note: u_out is binary (0 or 1). We want indices where u_out is 0.
    mask = u_out == 0

    if np.sum(mask) == 0:
        return 0.0

    # Calculate absolute error on masked elements
    abs_error = np.abs(preds[mask] - targets[mask])

    # Return mean
    return float(np.mean(abs_error))
