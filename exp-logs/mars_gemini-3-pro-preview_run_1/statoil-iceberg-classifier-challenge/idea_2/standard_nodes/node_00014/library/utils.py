import os
import random
import numpy as np
import torch
from library.config import SEED, DEVICE


def set_seed(seed=SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
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


def save_model(model, path):
    """
    Saves the model's state dictionary to the specified file path.
    Creates the parent directory if it does not exist.

    Args:
        model (torch.nn.Module): The model to save.
        path (str): The file path where the state dictionary will be saved.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    torch.save(model.state_dict(), path)


def load_model(model, path, device=DEVICE):
    """
    Loads the model's state dictionary from the specified file path.

    Args:
        model (torch.nn.Module): The model instance to load weights into.
        path (str): The file path of the saved state dictionary.
        device (str): The device to map the location to (e.g., 'cpu', 'cuda').
                      Defaults to DEVICE from config.

    Returns:
        torch.nn.Module: The model with loaded weights.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)

    return model
