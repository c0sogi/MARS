import os
import random
import copy
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_state_dict(model: torch.nn.Module, path: str) -> None:
    """
    Saves the model's state dictionary to a file using copy.deepcopy to ensure
    immutability of the captured state.

    Args:
        model (torch.nn.Module): The model to save.
        path (str): The file path where the state dict will be saved.
    """
    # Ensure the directory exists
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Deep copy the state dict to prevent reference issues.
    # This ensures we retain the exact parameters at this point in time,
    # protecting against any reference mutability if the model is updated later.
    state_dict = copy.deepcopy(model.state_dict())

    # Save to disk
    torch.save(state_dict, path)
