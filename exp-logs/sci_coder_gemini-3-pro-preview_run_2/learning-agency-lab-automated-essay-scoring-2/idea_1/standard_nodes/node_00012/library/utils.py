import os
import random
import numpy as np
import torch
import joblib
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # Torch seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_artifact(obj, path: str):
    """
    Serializes and saves a Python object (e.g., model, vectorizer) to the specified path using joblib.
    Ensures the parent directory exists before saving.

    Args:
        obj: The object to save.
        path (str): The file path where the object should be saved.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(path: str):
    """
    Loads a serialized Python object from the specified path using joblib.

    Args:
        path (str): The file path to load from.

    Returns:
        The loaded object.

    Raises:
        FileNotFoundError: If the file does not exist at the specified path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found at {path}")
    return joblib.load(path)
