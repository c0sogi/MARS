import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, Numpy, and Torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_rmse(predictions, targets):
    """
    Calculates the Root Mean Squared Error between predictions and targets.
    Supports both torch.Tensor and np.ndarray inputs.

    Args:
        predictions: Predicted pixel intensities.
        targets: Ground truth pixel intensities.

    Returns:
        float: The RMSE value.
    """
    # Convert tensors to numpy if necessary
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def save_checkpoint(model, optimizer, scheduler, epoch, score, filepath):
    """
    Saves the model checkpoint including optimizer and scheduler states.

    Args:
        model: The PyTorch model.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        epoch (int): Current epoch.
        score (float): Validation score (RMSE).
        filepath (str): Path to save the checkpoint.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "score": score,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(
    model, filepath, optimizer=None, scheduler=None, device=Config.DEVICE
):
    """
    Loads a model checkpoint.

    Args:
        model: The PyTorch model to load weights into.
        filepath (str): Path to the checkpoint file.
        optimizer: The optimizer to load state into (optional).
        scheduler: The scheduler to load state into (optional).
        device: Device to map the location to.

    Returns:
        tuple: (epoch, score) from the checkpoint.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if (
        scheduler
        and "scheduler_state_dict" in checkpoint
        and checkpoint["scheduler_state_dict"] is not None
    ):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    score = checkpoint.get("score", float("inf"))

    return epoch, score
