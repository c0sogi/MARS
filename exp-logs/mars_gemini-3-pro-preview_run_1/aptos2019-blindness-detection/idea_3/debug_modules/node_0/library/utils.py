import os
import random
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
import library.config as config


def seed_everything(seed=config.SEED):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def quadratic_weighted_kappa(y_true, y_pred):
    """
    Calculates the Quadratic Weighted Kappa (QWK) score.

    Args:
        y_true: Array-like of ground truth labels (integers 0-4).
        y_pred: Array-like of predicted labels (integers 0-4).

    Returns:
        float: The QWK score.
    """
    # Ensure inputs are numpy arrays of integers
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def save_checkpoint(model, optimizer, epoch, score, filename=config.MODEL_SAVE_PATH):
    """
    Saves the model checkpoint.

    Args:
        model: The PyTorch model to save.
        optimizer: The optimizer state.
        epoch: Current epoch number.
        score: Validation score (QWK) at this epoch.
        filename: Path to save the checkpoint.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "score": score,
    }
    torch.save(state, filename)


def load_checkpoint(model, optimizer=None, filename=config.MODEL_SAVE_PATH):
    """
    Loads a model checkpoint.

    Args:
        model: The PyTorch model to load weights into.
        optimizer: The optimizer to load state into (optional).
        filename: Path to the checkpoint file.

    Returns:
        dict: The loaded checkpoint dictionary if successful, else None.
    """
    if not os.path.exists(filename):
        return None

    checkpoint = torch.load(filename, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
