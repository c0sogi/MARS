import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=Config.SEED):
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


def calculate_roc_auc(y_true, y_score):
    """
    Calculates the Area Under the ROC Curve.

    Args:
        y_true (array-like): Ground truth labels (binary).
        y_score (array-like): Predicted probabilities.

    Returns:
        float: ROC AUC score.
    """
    # Ensure inputs are numpy arrays and flattened
    y_true = np.array(y_true).reshape(-1)
    y_score = np.array(y_score).reshape(-1)

    # Handle edge case where only one class is present to avoid ValueError
    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(y_true, y_score)


def save_checkpoint(model, optimizer, scheduler, epoch, score, filename):
    """
    Saves the model checkpoint to the configured output directory.

    Args:
        model: The PyTorch model.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        epoch (int): Current epoch.
        score (float): Validation score (ROC AUC).
        filename (str): Name of the file to save (e.g., 'model_fold_0.pth').
    """
    # Ensure output directory exists
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    save_path = os.path.join(Config.OUTPUT_DIR, filename)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "score": score,
    }

    torch.save(state, save_path)


def load_checkpoint(model, filename, optimizer=None, scheduler=None, device="cpu"):
    """
    Loads a model checkpoint from the configured output directory.

    Args:
        model: The PyTorch model instance.
        filename (str): Name of the file to load.
        optimizer: (Optional) Optimizer to load state into.
        scheduler: (Optional) Scheduler to load state into.
        device (str): Device to map location.

    Returns:
        dict: The full checkpoint dictionary.
    """
    load_path = os.path.join(Config.OUTPUT_DIR, filename)

    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint file not found: {load_path}")

    checkpoint = torch.load(load_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking losses and metrics during training.
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
