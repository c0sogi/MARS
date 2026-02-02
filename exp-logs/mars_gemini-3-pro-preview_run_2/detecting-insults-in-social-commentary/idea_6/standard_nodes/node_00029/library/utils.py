import os
import sys
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=Config.seed):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_score(y_true, y_pred):
    """
    Calculates the Area Under the Receiver Operating Curve (AUC).

    Args:
        y_true: Ground truth labels (binary).
        y_pred: Predicted probabilities.

    Returns:
        float: AUC score.
    """
    # Handle edge case where y_true has only one class (e.g., in very small batches or debug subsets)
    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(y_true, y_pred)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger:
    """
    Logs messages to both the console (stdout) and a specified log file.
    """

    def __init__(self, filename):
        self.filename = filename
        # Ensure the directory for the log file exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    def log(self, message):
        """
        Prints the message to console and appends it to the log file.
        """
        # Print to console
        print(message)
        # Append to file
        with open(self.filename, "a+", encoding="utf-8") as f:
            f.write(str(message) + "\n")


def save_checkpoint(state, filename):
    """
    Saves the training checkpoint (model state, optimizer, etc.) to a file.

    Args:
        state (dict): The state dictionary to save.
        filename (str): The path to save the checkpoint to.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(
    model, filename, optimizer=None, scheduler=None, device=Config.device
):
    """
    Loads a checkpoint into the model (and optionally optimizer/scheduler).

    Args:
        model: The PyTorch model to load weights into.
        filename (str): Path to the checkpoint file.
        optimizer (optional): Optimizer to load state into.
        scheduler (optional): Scheduler to load state into.
        device: Device to map the location to.

    Returns:
        dict: The loaded checkpoint dictionary, or None if file not found.
    """
    if not os.path.exists(filename):
        print(f"Checkpoint file not found: {filename}")
        return None

    checkpoint = torch.load(filename, map_location=device)

    # Determine if the checkpoint is a full dict or just state_dict
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Handle 'module.' prefix if the model was trained with DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    # Load optimizer state if available and requested
    if (
        optimizer is not None
        and isinstance(checkpoint, dict)
        and "optimizer" in checkpoint
    ):
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Load scheduler state if available and requested
    if (
        scheduler is not None
        and isinstance(checkpoint, dict)
        and "scheduler" in checkpoint
    ):
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint
