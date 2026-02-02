import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Force deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, scheduler, epoch, score, filepath):
    """
    Saves the model checkpoint including optimizer and scheduler states.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        epoch (int): Current epoch number.
        score (float): Validation score (e.g., loss or AUC) at this checkpoint.
        filepath (str): Path to save the checkpoint file.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "score": score,
    }
    torch.save(state, filepath)


def load_checkpoint(model, filepath, optimizer=None, scheduler=None, device="cpu"):
    """
    Loads a model checkpoint.

    Args:
        model (torch.nn.Module): The model to load weights into.
        filepath (str): Path to the checkpoint file.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler to load state into.
        device (str): Device to map the location to ('cpu' or 'cuda').

    Returns:
        tuple: (score, epoch) from the checkpoint.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if (
        scheduler is not None
        and "scheduler" in checkpoint
        and checkpoint["scheduler"] is not None
    ):
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint.get("score", None), checkpoint.get("epoch", 0)


def save_metrics(metrics: dict, filepath: str):
    """
    Appends a dictionary of metrics to a CSV file.

    Args:
        metrics (dict): Dictionary containing metric names and values.
        filepath (str): Path to the CSV file.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    df = pd.DataFrame([metrics])

    if os.path.exists(filepath):
        df.to_csv(filepath, mode="a", header=False, index=False)
    else:
        df.to_csv(filepath, mode="w", header=True, index=False)


def print_metrics(metrics: dict):
    """
    Prints metrics to the console with full precision.

    Args:
        metrics (dict): Dictionary containing metric names and values.
    """
    # Print without formatting to preserve full precision
    parts = [f"{k}: {v}" for k, v in metrics.items()]
    print(" | ".join(parts))
