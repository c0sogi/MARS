import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_roc_auc(y_true, y_scores):
    """
    Calculates the Area Under the ROC Curve.

    Args:
        y_true (array-like or Tensor): Ground truth binary labels.
        y_scores (array-like or Tensor): Predicted probabilities for the positive class.

    Returns:
        float: The ROC AUC score.
    """
    # Detach and move to CPU if inputs are tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.detach().cpu().numpy()

    # Handle potential edge case where batch has only one class
    try:
        return roc_auc_score(y_true, y_scores)
    except ValueError:
        # This can happen if y_true has only one class (all 0s or all 1s)
        # Return 0.5 as a neutral metric in this edge case
        return 0.5


def save_checkpoint(state, filename):
    """
    Saves the model checkpoint to the working directory defined in Config.

    Args:
        state (dict): State dictionary containing model parameters, optimizer state, etc.
        filename (str): Name of the file to save (e.g., 'model_seed_0.pth').
    """
    # Ensure directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    filepath = os.path.join(Config.WORKING_DIR, filename)
    torch.save(state, filepath)


def load_checkpoint(model, filename, optimizer=None, device=Config.DEVICE):
    """
    Loads the model checkpoint from the working directory.

    Args:
        model (torch.nn.Module): The model to load weights into.
        filename (str): Name of the file to load.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str): Device to map the checkpoint to.

    Returns:
        dict: The loaded checkpoint dictionary.
    """
    filepath = os.path.join(Config.WORKING_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    # Handle both full checkpoint dicts and direct state dicts
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
