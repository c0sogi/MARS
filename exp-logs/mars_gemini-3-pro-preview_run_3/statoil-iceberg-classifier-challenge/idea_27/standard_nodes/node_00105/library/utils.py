import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, filename):
    """
    Saves the model state dictionary to the working directory defined in Config.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        filename (str): Name of the file to save (e.g., 'model_fold_0.pth').
    """
    # Ensure the working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    filepath = os.path.join(Config.WORKING_DIR, filename)
    torch.save(state, filepath)


def load_checkpoint(filename, device=Config.DEVICE):
    """
    Loads the model state dictionary from the working directory defined in Config.

    Args:
        filename (str): Name of the file to load.
        device (str): Device to map the storage to ('cpu' or 'cuda').

    Returns:
        dict: The loaded state dictionary.
    """
    filepath = os.path.join(Config.WORKING_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at: {filepath}")

    # Load the checkpoint mapping to the specified device
    checkpoint = torch.load(filepath, map_location=device)
    return checkpoint
