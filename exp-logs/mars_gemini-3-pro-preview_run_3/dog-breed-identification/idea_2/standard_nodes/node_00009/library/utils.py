import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Saves the model checkpoint including model state, optimizer state, epoch, and loss.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state to save.
        epoch (int): The current training epoch.
        loss (float): The validation loss at this checkpoint.
        path (str): The file path to save the checkpoint to.
    """
    # Ensure the directory exists
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, path, device):
    """
    Loads a model checkpoint.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer): The optimizer to load state into.
        path (str): The file path of the checkpoint.
        device (str or torch.device): The device to map the location to.

    Returns:
        tuple: (epoch, loss) from the saved checkpoint.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at {path}")

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", float("inf"))

    return epoch, loss


def calculate_log_loss(y_true, y_pred, labels=None):
    """
    Calculates the Multi-Class Log Loss.

    Args:
        y_true (array-like): Ground truth (correct) labels. Can be class indices or one-hot encoded.
        y_pred (array-like): Predicted probabilities, as returned by a classifier's predict_proba method.
        labels (array-like, optional): List of labels to index the matrix. This may be used to reorder
                                       or select a subset of labels.

    Returns:
        float: The log loss.
    """
    # sklearn.metrics.log_loss handles both 1D class indices and 2D one-hot encoded targets for y_true
    return log_loss(y_true, y_pred, labels=labels)
