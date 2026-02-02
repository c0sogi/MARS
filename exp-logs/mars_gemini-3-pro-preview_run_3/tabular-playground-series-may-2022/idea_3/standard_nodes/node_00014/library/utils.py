import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Returns the appropriate PyTorch device ('cuda' or 'cpu').

    Returns:
        torch.device: The device object.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
):
    """
    Saves the model checkpoint including model state, optimizer state, epoch, and loss.
    Automatically creates the directory if it does not exist.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state to save (can be None).
        epoch (int): The current training epoch.
        loss (float): The current loss value.
        path (str): The file path to save the checkpoint to.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    state = {"model_state_dict": model.state_dict(), "epoch": epoch, "loss": loss}

    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: torch.device = None,
):
    """
    Loads a model checkpoint from the specified path.

    Args:
        path (str): The file path of the checkpoint.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into. Defaults to None.
        device (torch.device, optional): The device to map the location to. Defaults to None (auto-detect).

    Returns:
        dict: The full checkpoint dictionary containing epoch, loss, etc.
    """
    if device is None:
        device = get_device()

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at {path}")

    # weights_only=True is recommended for security in newer PyTorch versions
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def compute_auc(y_true, y_pred) -> float:
    """
    Computes the Area Under the ROC Curve (AUC).
    Handles both PyTorch Tensors and NumPy arrays.

    Args:
        y_true: Ground truth labels (binary).
        y_pred: Predicted probabilities.

    Returns:
        float: The ROC AUC score.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    return roc_auc_score(y_true, y_pred)
