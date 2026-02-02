import os
import random
import numpy as np
import torch
import sys
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across various libraries.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_auc(y_true, y_pred):
    """
    Computes the Area Under the ROC Curve (ROC AUC).

    Args:
        y_true (np.ndarray): Ground truth binary labels (N, num_classes).
        y_pred (np.ndarray): Predicted probabilities (N, num_classes).

    Returns:
        float: Macro-averaged ROC AUC score.
    """
    try:
        # 'macro' calculates metrics for each label, and finds their unweighted mean.
        # This does not take label imbalance into account.
        score = roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        # Handle cases where a class might not be present in the batch
        score = 0.0
    return score


def save_checkpoint(state, filename="checkpoint.pth"):
    """
    Saves the training state to a file.

    Args:
        state (dict): Dictionary containing model state, optimizer state, epoch, etc.
        filename (str): Name of the file to save.
    """
    filepath = os.path.join(Config.WORKING_DIR, filename)
    torch.save(state, filepath)
    # Silent save, no print needed as per instructions to avoid clutter unless necessary


def load_checkpoint(model, filename, optimizer=None, device=Config.DEVICE):
    """
    Loads a checkpoint into the model and optionally the optimizer.

    Args:
        model (torch.nn.Module): The model to load weights into.
        filename (str): The filename of the checkpoint in WORKING_DIR.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str): Device to map the location to.

    Returns:
        dict: The full checkpoint dictionary (useful for retrieving epoch/score).
    """
    filepath = os.path.join(Config.WORKING_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    # Load model state
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Assume the checkpoint is just the state dict
        model.load_state_dict(checkpoint)

    # Load optimizer state if provided
    if optimizer is not None:
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        elif "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


class Logger:
    """
    Simple logger that writes to a file and stdout.
    """

    def __init__(self, filename="log.txt"):
        self.filepath = os.path.join(Config.WORKING_DIR, filename)
        # Create/Clear the log file
        with open(self.filepath, "w") as f:
            pass

    def log(self, message):
        """
        Writes a message to the log file and prints it to stdout.
        """
        print(message)
        with open(self.filepath, "a") as f:
            f.write(str(message) + "\n")
