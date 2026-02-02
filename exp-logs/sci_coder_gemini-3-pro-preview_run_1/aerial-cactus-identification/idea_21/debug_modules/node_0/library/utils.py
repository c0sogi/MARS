import os
import random
import shutil
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import CHECKPOINTS_DIR


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training loops.
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


def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    """
    Saves the model state to the checkpoints directory.

    Args:
        state (dict): The state dictionary containing model parameters, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): The filename for the checkpoint.
    """
    # Ensure the directory exists
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    filepath = os.path.join(CHECKPOINTS_DIR, filename)
    torch.save(state, filepath)

    if is_best:
        # Create a 'best_' prefixed version of the filename
        best_filename = f"best_{filename}"
        best_filepath = os.path.join(CHECKPOINTS_DIR, best_filename)
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(filename, model, optimizer=None):
    """
    Loads a checkpoint into the model and optional optimizer.

    Args:
        filename (str): Path to the checkpoint file or filename within CHECKPOINTS_DIR.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.

    Returns:
        epoch (int): The epoch saved in the checkpoint (default 0).
        best_score (float): The best score saved in the checkpoint (default 0.0).
    """
    # Check if filename is a path or just a name
    if not os.path.isfile(filename):
        filepath = os.path.join(CHECKPOINTS_DIR, filename)
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f"Checkpoint file not found: {filename} or {filepath}"
            )
        filename = filepath

    # Load on CPU to avoid GPU OOM or device mismatch; caller moves to device
    checkpoint = torch.load(filename, map_location="cpu")

    # Handle state dict
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Handle DataParallel keys (remove 'module.' prefix if present)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)

    # Load optimizer state if provided
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    epoch = checkpoint.get("epoch", 0)
    best_score = checkpoint.get("best_score", 0.0)

    return epoch, best_score


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve.

    Args:
        y_true (array-like): True binary labels.
        y_pred (array-like): Target scores (probability estimates).

    Returns:
        float: ROC AUC score.
    """
    # Convert tensors to numpy if necessary
    if hasattr(y_true, "cpu"):
        y_true = y_true.detach().cpu().numpy()
    if hasattr(y_pred, "cpu"):
        y_pred = y_pred.detach().cpu().numpy()

    # Handle edge case where only one class is present
    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(y_true, y_pred)
