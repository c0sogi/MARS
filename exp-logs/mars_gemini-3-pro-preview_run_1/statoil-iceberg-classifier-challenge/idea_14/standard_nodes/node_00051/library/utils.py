import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
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

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model, path):
    """
    Saves the model state dictionary to the specified path.
    Ensures the parent directory exists before saving.

    Args:
        model (torch.nn.Module): The model instance to save.
        path (str): The target file path.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    torch.save(model.state_dict(), path)


def load_model(model, path, device=Config.DEVICE):
    """
    Loads the model state dictionary from the specified path.

    Args:
        model (torch.nn.Module): The model instance to load weights into.
        path (str): The file path of the checkpoint.
        device (str): The device to map the weights to. Defaults to Config.DEVICE.

    Returns:
        torch.nn.Module: The model with loaded weights.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model checkpoint not found at: {path}")

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def log_message(message):
    """
    Logs a general message to stdout.

    Args:
        message (str): The message to log.
    """
    print(message)


def log_metrics(epoch, metrics):
    """
    Logs training/validation metrics for a specific epoch.
    Prints values with full precision.

    Args:
        epoch (int): The current epoch number.
        metrics (dict): A dictionary where keys are metric names and values are floats.
    """
    # Construct string with full precision for floats
    parts = []
    for k, v in metrics.items():
        parts.append(f"{k}: {v}")

    print(f"Epoch {epoch} | " + " | ".join(parts))
