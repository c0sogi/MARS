import os
import random
import re
import numpy as np
import torch
from sklearn.metrics import f1_score
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_text(text: str) -> str:
    """
    Cleans and normalizes the input text by removing HTML tags, special characters,
    and extra whitespace.

    Args:
        text (str): The raw input string.

    Returns:
        str: The cleaned and normalized string.
    """
    if not isinstance(text, str):
        return ""

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove non-alphanumeric characters (keep spaces)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Convert to lowercase
    text = text.lower()

    # Normalize whitespace (replace tabs, newlines, and multi-spaces with single space)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def calculate_f1_score(y_true, y_pred, average="micro"):
    """
    Calculates the F1 score for multi-label classification.
    Handles both PyTorch tensors and NumPy arrays.

    Args:
        y_true (Union[np.ndarray, torch.Tensor]): Ground truth binary labels.
        y_pred (Union[np.ndarray, torch.Tensor]): Predicted binary labels.
        average (str): The averaging method for F1 score ('micro', 'macro', 'samples', etc.).

    Returns:
        float: The calculated F1 score.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    return f1_score(y_true, y_pred, average=average, zero_division=0)


def save_checkpoint(model, optimizer, epoch, loss, path=Config.MODEL_SAVE_PATH):
    """
    Saves the model and optimizer state to a file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch number.
        loss (float): The current validation loss.
        path (str): The file path to save the checkpoint.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "loss": loss,
    }

    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, device=Config.DEVICE):
    """
    Loads the model and optimizer state from a file.

    Args:
        path (str): The file path to load the checkpoint from.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str): The device to map the location to (e.g., 'cuda', 'cpu').

    Returns:
        dict: The loaded checkpoint dictionary if successful, None otherwise.
    """
    if not os.path.exists(path):
        return None

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
