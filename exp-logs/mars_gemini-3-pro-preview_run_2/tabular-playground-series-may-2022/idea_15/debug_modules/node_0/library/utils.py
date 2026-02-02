import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed: int):
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
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Deterministic operations for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_auc(y_true, y_pred):
    """
    Computes the Area Under the ROC Curve.

    Args:
        y_true: Ground truth labels (binary). Can be numpy array or torch Tensor.
        y_pred: Predicted probabilities. Can be numpy array or torch Tensor.

    Returns:
        float: The AUC score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Handle potential NaN values in predictions by replacing them with 0.5 (neutral)
    # or raising an error depending on strictness. Here we assume clean output from model.
    return roc_auc_score(y_true, y_pred)


def save_checkpoint(model, optimizer, scheduler, epoch, score, filename):
    """
    Saves the model checkpoint including optimizer and scheduler states.

    Args:
        model: The PyTorch model.
        optimizer: The optimizer used during training.
        scheduler: The learning rate scheduler.
        epoch (int): Current epoch number.
        score (float): Validation score (AUC).
        filename (str): Path to save the checkpoint file.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "score": score,
    }

    torch.save(checkpoint, filename)


def load_checkpoint(filename, model, optimizer=None, scheduler=None, device="cpu"):
    """
    Loads a model checkpoint.

    Args:
        filename (str): Path to the checkpoint file.
        model: The PyTorch model to load weights into.
        optimizer: The optimizer to load state into (optional).
        scheduler: The scheduler to load state into (optional).
        device (str): Device to map the location to ('cpu' or 'cuda').

    Returns:
        dict: The checkpoint dictionary containing metadata like epoch and score.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found at {filename}")

    checkpoint = torch.load(filename, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
