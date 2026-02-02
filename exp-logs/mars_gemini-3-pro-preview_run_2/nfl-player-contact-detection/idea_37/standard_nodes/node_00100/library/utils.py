import os
import random
import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef


def seed_everything(seed: int = 42):
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_mcc(y_true, y_pred):
    """
    Calculates the Matthews Correlation Coefficient (MCC).
    Handles PyTorch tensors by detaching and converting to NumPy arrays.

    Args:
        y_true: Ground truth labels (Tensor or array-like).
        y_pred: Predicted labels (Tensor or array-like).

    Returns:
        float: The MCC score.
    """
    # Convert tensors to numpy if necessary
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are integers (binary labels)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    return matthews_corrcoef(y_true, y_pred)


def save_checkpoint(model, optimizer, epoch, score, path):
    """
    Saves the model and optimizer state to a checkpoint file.

    Args:
        model: The PyTorch model.
        optimizer: The optimizer.
        epoch (int): Current epoch number.
        score (float): Validation score (MCC).
        path (str): Path to save the checkpoint.
    """
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "score": score,
    }
    torch.save(state, path)


def load_checkpoint(model, path, optimizer=None, device="cpu"):
    """
    Loads a model checkpoint.

    Args:
        model: The PyTorch model to load weights into.
        path (str): Path to the checkpoint file.
        optimizer: The optimizer to load state into (optional).
        device (str): Device to map the checkpoint to ('cpu' or 'cuda').

    Returns:
        dict: The full checkpoint dictionary containing epoch and score.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
