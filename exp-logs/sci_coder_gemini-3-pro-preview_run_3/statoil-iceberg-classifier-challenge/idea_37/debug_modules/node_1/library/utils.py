import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set generator for DataLoader if needed (though usually handled by worker_init_fn)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Saves the model checkpoint including optimizer state and metrics.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state to save.
        epoch (int): The current epoch number.
        loss (float): The validation loss at this checkpoint.
        path (str): The file path to save the checkpoint to.
    """
    # Ensure the directory exists
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "loss": loss,
    }

    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, device=None):
    """
    Loads a model checkpoint.

    Args:
        path (str): The file path to load the checkpoint from.
        model (torch.nn.Module): The model instance to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str, optional): The device to map the checkpoint to (e.g., 'cuda', 'cpu').
                                Defaults to Config.DEVICE if None.

    Returns:
        dict: The loaded checkpoint dictionary containing epoch and loss info.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    if device is None:
        device = Config.DEVICE

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
