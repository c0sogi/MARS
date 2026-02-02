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
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, path: str):
    """
    Saves the model state dictionary to the specified path.
    Ensures the directory exists before saving.

    Args:
        model (torch.nn.Module): The model to save.
        path (str): The file path to save the checkpoint to.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path: str, device: torch.device = None):
    """
    Loads the model state dictionary from the specified path.

    Args:
        model (torch.nn.Module): The model to load weights into.
        path (str): The file path of the checkpoint.
        device (torch.device, optional): The device to load the model onto.
                                         Defaults to CUDA if available, else CPU.

    Returns:
        model (torch.nn.Module): The model with loaded weights.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model
