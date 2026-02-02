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
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic operations for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_roc_auc(y_true, y_pred):
    """
    Computes the Area Under the Receiver Operating Characteristic Curve (ROC AUC).
    Handles both NumPy arrays and PyTorch tensors.

    Args:
        y_true (array-like or torch.Tensor): True binary labels.
        y_pred (array-like or torch.Tensor): Target scores (probability estimates).

    Returns:
        float: The ROC AUC score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    return roc_auc_score(y_true, y_pred)


def save_checkpoint(state, filename):
    """
    Saves the model checkpoint to the configured checkpoint directory.

    Args:
        state (dict): State dictionary containing model parameters, optimizer state, etc.
        filename (str): Name of the file to save (e.g., 'model_seed_0.pth').
    """
    filepath = os.path.join(Config.CHECKPOINT_DIR, filename)
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(filename, model, optimizer=None, device=Config.DEVICE):
    """
    Loads a model checkpoint from the configured checkpoint directory.

    Args:
        filename (str): Name of the file to load.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str): Device to map the location to.

    Returns:
        dict: The loaded checkpoint dictionary (useful for retrieving epoch, score, etc.).
    """
    filepath = os.path.join(Config.CHECKPOINT_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
