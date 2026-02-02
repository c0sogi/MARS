import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed: int = Config.SEED) -> None:
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Returns the device to be used for training/inference.

    Returns:
        torch.device: 'cuda' if available, else 'cpu'.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_metrics(epoch, epochs, train_loss, train_auc, val_loss, val_auc):
    """
    Prints training and validation metrics with full precision as required.

    Args:
        epoch (int): Current epoch number.
        epochs (int): Total number of epochs.
        train_loss (float): Training loss.
        train_auc (float): Training AUC.
        val_loss (float): Validation loss.
        val_auc (float): Validation AUC.
    """
    print(
        f"Epoch {epoch}/{epochs} - Train Loss: {train_loss} AUC: {train_auc} | Val Loss: {val_loss} AUC: {val_auc}"
    )
