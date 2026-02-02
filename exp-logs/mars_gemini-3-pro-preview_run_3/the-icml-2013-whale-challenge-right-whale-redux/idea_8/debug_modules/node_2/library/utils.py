import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_score(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve (AUC).

    Args:
        y_true (np.array or list): Ground truth labels.
        y_pred (np.array or list): Predicted probabilities.

    Returns:
        float: The ROC AUC score.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Fallback if only one class is present in the provided set
    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(y_true, y_pred)


def save_checkpoint(model, optimizer, scheduler, epoch, score, filename):
    """
    Saves the model state and training metadata to a file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        scheduler (object): The learning rate scheduler state.
        epoch (int): The current training epoch.
        score (float): The validation score (AUC) at this checkpoint.
        filename (str): The path where the checkpoint will be saved.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "score": score,
    }
    torch.save(state, filename)


def load_checkpoint(
    model, filename, optimizer=None, scheduler=None, device=Config.DEVICE
):
    """
    Loads a checkpoint into the model, optimizer, and scheduler.

    Args:
        model (torch.nn.Module): The model instance to load weights into.
        filename (str): Path to the checkpoint file.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (object, optional): Scheduler to load state into.
        device (str): Device to map the checkpoint to (default: Config.DEVICE).

    Returns:
        dict: The full checkpoint dictionary containing metadata (epoch, score).
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found at {filename}")

    checkpoint = torch.load(filename, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
