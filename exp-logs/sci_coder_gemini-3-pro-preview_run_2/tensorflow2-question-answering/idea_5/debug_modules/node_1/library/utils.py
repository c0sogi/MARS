import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to {seed}")


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Saves the model and optimizer state to a checkpoint file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch number.
        loss (float): The validation loss at this checkpoint.
        path (str): The file path to save the checkpoint.
    """
    # Ensure the directory exists
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "loss": loss,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: Epoch {epoch}, Loss {loss} -> {path}")


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    """
    Loads a model checkpoint.

    Args:
        path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str): The device to map the checkpoint to ('cpu' or 'cuda').

    Returns:
        dict: The loaded checkpoint dictionary containing epoch, loss, etc., or None if not found.
    """
    if not os.path.exists(path):
        print(f"Checkpoint file not found at {path}")
        return None

    print(f"Loading checkpoint from {path} to {device}...")
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", -1)
    loss = checkpoint.get("loss", float("inf"))

    print(f"Successfully loaded checkpoint from Epoch {epoch} with Loss {loss}")

    return checkpoint
