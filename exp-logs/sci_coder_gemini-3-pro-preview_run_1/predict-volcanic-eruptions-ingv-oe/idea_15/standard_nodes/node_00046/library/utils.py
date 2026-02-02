import os
import random
import pickle
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error


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
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mae_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Mean Absolute Error (MAE) between true and predicted values.

    Args:
        y_true (np.ndarray): Ground truth target values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: The MAE score.
    """
    return mean_absolute_error(y_true, y_pred)


def save_pickle(obj, path: str):
    """
    Saves a Python object to a file using pickle.
    Ensures the directory exists before saving.

    Args:
        obj: The Python object to save.
        path (str): The file path where the object should be saved.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    """
    Loads a Python object from a pickle file.

    Args:
        path (str): The file path to load from.

    Returns:
        The loaded Python object.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def get_device() -> torch.device:
    """
    Selects the available hardware device (GPU if available, else CPU).

    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
