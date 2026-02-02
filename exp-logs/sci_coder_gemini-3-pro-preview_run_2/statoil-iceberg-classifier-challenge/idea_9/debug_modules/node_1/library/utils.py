import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

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
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, epoch, loss, filename):
    """
    Saves the model and optimizer state to a file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch number.
        loss (float): The validation loss at this checkpoint.
        filename (str): The path where the checkpoint will be saved.
    """
    # Ensure the directory exists
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)

    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "loss": loss,
    }
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None):
    """
    Loads the model and optimizer state from a file.

    Args:
        filename (str): The path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.

    Returns:
        dict: The full checkpoint dictionary containing epoch, loss, etc.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    # Load checkpoint to the configured device
    checkpoint = torch.load(filename, map_location=Config.DEVICE)

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint


def log_metrics(epoch, train_loss, val_loss, val_acc=None, time_elapsed=None):
    """
    Formats and prints training and validation metrics with full precision.

    Args:
        epoch (int): Current epoch.
        train_loss (float): Training loss.
        val_loss (float): Validation loss.
        val_acc (float, optional): Validation accuracy.
        time_elapsed (float, optional): Time taken for the epoch.
    """
    msg = f"Epoch {epoch} | Train Loss: {train_loss} | Val Loss: {val_loss}"

    if val_acc is not None:
        msg += f" | Val Acc: {val_acc}"

    if time_elapsed is not None:
        msg += f" | Time: {time_elapsed}"

    print(msg)
