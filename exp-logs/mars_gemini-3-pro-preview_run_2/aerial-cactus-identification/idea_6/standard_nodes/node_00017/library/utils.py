import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import DEVICE


def seed_everything(seed: int):
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve.

    Args:
        y_true: Ground truth labels (Tensor, numpy array, or list).
        y_pred: Predicted probabilities (Tensor, numpy array, or list).

    Returns:
        float: The ROC AUC score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays and flattened
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        # This can happen if y_true has only one class present in the batch/set
        return 0.5


def save_checkpoint(state, filename):
    """
    Saves the model checkpoint to the specified filename.

    Args:
        state (dict): The state dictionary to save (e.g., model weights, optimizer, epoch).
        filename (str): The path where the checkpoint will be saved.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None):
    """
    Loads a model checkpoint from the specified filename.

    Args:
        filename (str): The path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.

    Returns:
        epoch (int): The epoch saved in the checkpoint (default 0).
        best_score (float): The best score saved in the checkpoint (default None).
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    # Load checkpoint to the configured device
    checkpoint = torch.load(filename, map_location=DEVICE)

    # Load model weights
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # Fallback if the file only contains the state dict directly
        model.load_state_dict(checkpoint)

    # Load optimizer state if provided
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    epoch = checkpoint.get("epoch", 0)
    best_score = checkpoint.get("best_score", None)

    return epoch, best_score
