import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

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


def get_device() -> torch.device:
    """
    Returns the appropriate torch device based on availability.

    Returns:
        torch.device: 'cuda' if available, otherwise 'cpu'.
    """
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_str)


def save_checkpoint(model, optimizer, epoch, loss, filename=None):
    """
    Saves the model and optimizer state to a checkpoint file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch number.
        loss (float): The validation loss at this checkpoint.
        filename (str, optional): Path to save the checkpoint.
                                  Defaults to Config.MODEL_CHECKPOINT_PATH.
    """
    if filename is None:
        filename = Config.MODEL_CHECKPOINT_PATH

    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "loss": loss,
    }

    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(model, optimizer=None, filename=None, device=None):
    """
    Loads the model and optimizer state from a checkpoint file.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        filename (str, optional): Path to the checkpoint file.
                                  Defaults to Config.MODEL_CHECKPOINT_PATH.
        device (torch.device, optional): Device to map the checkpoint to.
                                         Defaults to current device.

    Returns:
        dict: The checkpoint dictionary containing epoch, loss, etc., or None if not found.
    """
    if filename is None:
        filename = Config.MODEL_CHECKPOINT_PATH

    if not os.path.exists(filename):
        print(f"No checkpoint found at {filename}")
        return None

    if device is None:
        device = get_device()

    checkpoint = torch.load(filename, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(
        f"Checkpoint loaded from {filename} (Epoch {checkpoint.get('epoch', 'Unknown')})"
    )
    return checkpoint


def print_metrics(metrics_dict):
    """
    Prints metric values with full precision (no rounding).

    Args:
        metrics_dict (dict): Dictionary where keys are metric names and values are scores.
    """
    output_parts = []
    for key, value in metrics_dict.items():
        output_parts.append(f"{key}: {value}")

    print(", ".join(output_parts))
