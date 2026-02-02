import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import SEED, DEVICE


def seed_everything(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
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


def get_device():
    """
    Returns the PyTorch device (CPU or CUDA) defined in the configuration.
    """
    return torch.device(DEVICE)


def save_checkpoint(state, filename):
    """
    Saves the model state dictionary and other training artifacts to a file.

    Args:
        state (dict): Dictionary containing model_state_dict, optimizer_state_dict, epoch, etc.
        filename (str): Path to save the checkpoint.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None):
    """
    Loads a checkpoint into the model and optionally the optimizer.

    Args:
        filename (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.

    Returns:
        dict: The loaded checkpoint dictionary (useful for retrieving epoch or best score).
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    checkpoint = torch.load(filename, map_location=get_device())

    # Load model weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Load optimizer state if provided and available
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve.

    Args:
        y_true (np.array): Ground truth labels (0 or 1).
        y_pred (np.array): Predicted probabilities.

    Returns:
        float: ROC AUC score. Returns 0.5 if only one class is present in y_true.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Check if we have both classes to calculate ROC AUC
    if len(np.unique(y_true)) < 2:
        return 0.5

    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return 0.5


def print_metric(name, value):
    """
    Prints a metric with full precision.
    """
    print(f"{name}: {value}")
