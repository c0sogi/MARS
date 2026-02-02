import os
import random
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across all libraries.
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
    Calculates the Quadratic Weighted Kappa metric.

    Args:
        y_true: Array-like of ground truth integer labels (0-4).
        y_pred: Array-like of predicted scores. Can be continuous floats.

    Returns:
        float: The quadratic weighted kappa score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Handle regression outputs: Clip to valid range [0, 4] and round to nearest integer
    if y_pred.dtype.kind in "fc":  # float or complex
        y_pred = np.round(y_pred.clip(0, 4)).astype(int)

    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def save_checkpoint(model, optimizer, epoch, score, path):
    """
    Saves the model state, optimizer state, and current metrics to a file.

    Args:
        model: The PyTorch model.
        optimizer: The optimizer (optional).
        epoch: Current epoch number.
        score: Validation score (QWK).
        path: Destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "score": score,
    }

    torch.save(state, path)


def load_checkpoint(model, path, optimizer=None, device=Config.DEVICE):
    """
    Loads a checkpoint into the model and optimizer.

    Args:
        model: The PyTorch model to load weights into.
        path: Path to the checkpoint file.
        optimizer: The optimizer to load state into (optional).
        device: Device to map the location to.

    Returns:
        tuple: (epoch, score) from the checkpoint.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    score = checkpoint.get("score", 0.0)

    return epoch, score


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
