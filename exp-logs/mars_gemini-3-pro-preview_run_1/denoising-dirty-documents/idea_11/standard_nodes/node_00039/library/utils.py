import os
import random
import numpy as np
import torch
import torch.nn.functional as F


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducible results.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def invert_signal(x: torch.Tensor) -> torch.Tensor:
    """
    Inverts the pixel intensity signal (1.0 - x).
    Used to map white background (1.0) to 0.0 for zero-padding compatibility.

    Args:
        x (torch.Tensor): Input tensor with values in [0, 1].

    Returns:
        torch.Tensor: Inverted tensor.
    """
    return 1.0 - x


def revert_signal(x: torch.Tensor) -> torch.Tensor:
    """
    Reverts the inverted signal back to the original intensity space (1.0 - x).

    Args:
        x (torch.Tensor): Input tensor with values in [0, 1].

    Returns:
        torch.Tensor: Reverted tensor.
    """
    return 1.0 - x


def rmse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the Root Mean Squared Error (RMSE) between predictions and targets.

    Args:
        pred (torch.Tensor): Predicted values.
        target (torch.Tensor): Ground truth values.

    Returns:
        torch.Tensor: Scalar RMSE loss.
    """
    return torch.sqrt(F.mse_loss(pred, target))


def save_checkpoint(state: dict, filename: str):
    """
    Saves the model training checkpoint to a file.
    Ensures the directory exists before saving.

    Args:
        state (dict): State dictionary containing model parameters, optimizer state, etc.
        filename (str): Path to save the checkpoint.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(
    filename: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None
) -> dict:
    """
    Loads a model checkpoint from a file.

    Args:
        filename (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.

    Returns:
        dict: The loaded checkpoint dictionary (useful for retrieving epoch, loss, etc.).
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    # Load to CPU first to avoid OOM or device mismatch, then let caller move model
    checkpoint = torch.load(filename, map_location="cpu")

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint
