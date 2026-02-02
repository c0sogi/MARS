import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
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


def save_checkpoint(state, filename):
    """
    Saves the model state (or any other object) to a file.
    Automatically creates the parent directory if it does not exist.

    Args:
        state (dict): The object to save (typically model.state_dict() or a dict containing it).
        filename (str): The path where the checkpoint will be saved.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(model, filename, device):
    """
    Loads a checkpoint into a model. Handles both direct state_dict saves
    and dictionary wrappers containing a 'state_dict' key.

    Args:
        model (torch.nn.Module): The model instance to load weights into.
        filename (str): The path to the checkpoint file.
        device (str or torch.device): The device to map the checkpoint to.

    Returns:
        model: The model with loaded weights.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found at: {filename}")

    # Load checkpoint to the specific device
    checkpoint = torch.load(filename, map_location=device)

    # Handle case where checkpoint is a dict containing state_dict (e.g., with optimizer state)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # Handle case where checkpoint is just the state_dict
        model.load_state_dict(checkpoint)

    return model
