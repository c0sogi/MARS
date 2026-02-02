import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed: int):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model: torch.nn.Module, path: str):
    """
    Saves the model's state dictionary to the specified path.
    Creates the directory if it does not exist.

    Args:
        model (torch.nn.Module): The model to save.
        path (str): The file path to save the checkpoint to.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: torch.nn.Module, path: str, device: torch.device):
    """
    Loads the model's state dictionary from the specified path.

    Args:
        model (torch.nn.Module): The model to load weights into.
        path (str): The file path of the checkpoint.
        device (torch.device): The device to map the weights to.

    Returns:
        torch.nn.Module: The model with loaded weights.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def calculate_roc_auc(y_true, y_scores):
    """
    Calculates the Area Under the Receiver Operating Characteristic Curve (ROC AUC).

    Args:
        y_true (array-like): True binary labels.
        y_scores (array-like): Target scores (probability estimates).

    Returns:
        float: The ROC AUC score. Returns 0.5 if only one class is present in y_true.
    """
    try:
        # ROC AUC requires both classes to be present
        if len(np.unique(y_true)) < 2:
            return 0.5
        return roc_auc_score(y_true, y_scores)
    except Exception:
        return 0.5
