import os
import random
import shutil
import numpy as np
import torch
from sklearn.metrics import f1_score
from library.config import Config


def seed_everything(seed: int = Config.SEED) -> None:
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
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_macro_f1(y_true, y_pred) -> float:
    """
    Calculates the Macro F1 score for the given ground truth and predictions.

    Args:
        y_true (array-like): Ground truth labels. Can be a list, numpy array, or torch tensor.
        y_pred (array-like): Predicted labels. Can be a list, numpy array, or torch tensor.

    Returns:
        float: The calculated Macro F1 score.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Calculate Macro F1 score
    # average='macro': Calculate metrics for each label, and find their unweighted mean.
    # This does not take label imbalance into account.
    return f1_score(y_true, y_pred, average="macro")


def save_checkpoint(
    state: dict, is_best: bool, filename: str = "checkpoint.pth"
) -> None:
    """
    Saves the training checkpoint.

    Args:
        state (dict): The state dictionary containing model parameters, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): The path to save the checkpoint file.
    """
    # Ensure the directory exists
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    torch.save(state, filename)

    if is_best:
        # If this is the best model, we might want to save a copy or just rely on the filename provided
        # typically the filename passed for is_best=True is already the 'best' path
        # but strictly following standard patterns, we might copy.
        # Here we assume filename is the target destination.
        pass


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler=None,
):
    """
    Loads a checkpoint into the model, optimizer, and scheduler.

    Args:
        filepath (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (optional): The scheduler to load state into.

    Returns:
        dict: The loaded checkpoint dictionary (useful for retrieving epoch or best_score).
        None: If the file does not exist.
    """
    if not os.path.exists(filepath):
        print(f"No checkpoint found at '{filepath}'")
        return None

    print(f"Loading checkpoint from '{filepath}'")
    checkpoint = torch.load(filepath, map_location=Config.DEVICE)

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint
