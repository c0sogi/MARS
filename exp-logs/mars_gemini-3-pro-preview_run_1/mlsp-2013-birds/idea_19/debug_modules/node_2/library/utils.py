import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, is_best, filepath):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model parameters, optimizer state, epoch, etc.
        is_best (bool): If True, saves a copy of the checkpoint as 'model_best.pth'.
        filepath (str): Full path to save the checkpoint file.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save the checkpoint
    torch.save(state, filepath)

    # If this is the best model so far, create a copy
    if is_best:
        best_filepath = os.path.join(os.path.dirname(filepath), "model_best.pth")
        torch.save(state, best_filepath)


def load_checkpoint(
    model, filepath, optimizer=None, scheduler=None, device=Config.DEVICE
):
    """
    Loads a checkpoint into the model, and optionally into the optimizer and scheduler.

    Args:
        model (torch.nn.Module): The model to load weights into.
        filepath (str): Path to the checkpoint file.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler to load state into.
        device (str): Device to map the location to.

    Returns:
        tuple: (start_epoch, best_score)
            start_epoch (int): The epoch to resume from (0 if not found).
            best_score (float): The best metric score recorded (0.0 if not found).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    # Initialize defaults
    start_epoch = 0
    best_score = 0.0

    # Check if the checkpoint is a dict with metadata or just a state_dict
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])

        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1

        if "best_score" in checkpoint:
            best_score = checkpoint["best_score"]

        if optimizer and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

        if scheduler and "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
    else:
        # Assume it's a raw state_dict
        model.load_state_dict(checkpoint)

    return start_epoch, best_score


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the macro-averaged ROC AUC score.

    Args:
        y_true (np.ndarray): Ground truth binary labels (N, NumClasses).
        y_pred (np.ndarray): Predicted probabilities (N, NumClasses).

    Returns:
        float: Macro-averaged ROC AUC score.
    """
    # Handle cases where a class might not be present in the provided set
    # sklearn's roc_auc_score throws an error if a class has only one label (all 0 or all 1)
    try:
        score = roc_auc_score(y_true, y_pred, average="macro")
        if np.isnan(score):
            raise ValueError("Global AUC is NaN")
    except ValueError:
        # Fallback: calculate per-class and average valid scores
        n_classes = y_true.shape[1]
        scores = []
        for i in range(n_classes):
            # Check if class exists in y_true
            if len(np.unique(y_true[:, i])) > 1:
                try:
                    s = roc_auc_score(y_true[:, i], y_pred[:, i])
                    scores.append(s)
                except ValueError:
                    pass

        if len(scores) > 0:
            score = np.mean(scores)
        else:
            score = 0.5  # Default to random guess if calculation fails completely

    return score


class AverageMeter:
    """Computes and stores the average and current value."""

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
