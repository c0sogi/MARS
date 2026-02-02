import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library import config


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
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve (ROC AUC).

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth binary labels.
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities.

    Returns:
        float: The ROC AUC score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    return roc_auc_score(y_true, y_pred)


def save_checkpoint(model, optimizer, epoch, score, filename):
    """
    Saves the model and optimizer state to a checkpoint file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch number.
        score (float): The validation score (e.g., ROC AUC) associated with this checkpoint.
        filename (str): The name of the file to save (e.g., 'best_model.pth').
    """
    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    filepath = os.path.join(config.WORKING_DIR, filename)

    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "score": score,
    }

    torch.save(state, filepath)


def load_checkpoint(model, filename, optimizer=None, device=config.DEVICE):
    """
    Loads a model checkpoint.

    Args:
        model (torch.nn.Module): The model instance to load weights into.
        filename (str): The filename of the checkpoint to load.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str): Device to map the location to.

    Returns:
        dict: The full checkpoint dictionary (containing epoch, score, etc.).
    """
    filepath = os.path.join(config.WORKING_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None and checkpoint["optimizer"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and accuracy during training.
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
