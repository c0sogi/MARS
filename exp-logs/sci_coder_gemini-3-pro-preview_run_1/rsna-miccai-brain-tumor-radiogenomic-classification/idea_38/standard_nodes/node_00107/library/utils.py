import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import seed_everything, DEVICE


def get_device():
    """
    Returns the PyTorch device configured in the library.
    """
    return torch.device(DEVICE)


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


def get_score(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve (ROC AUC).
    Handles edge cases where only one class is present in the batch.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ROC AUC requires at least one example of each class
    if len(np.unique(y_true)) < 2:
        return 0.5

    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return 0.5


def save_checkpoint(state, filename):
    """
    Saves the model checkpoint to the specified file.
    """
    torch.save(state, filename)


def load_checkpoint(model, filename, device=None):
    """
    Loads the model checkpoint from the specified file.
    Returns the model and the best score recorded in the checkpoint.
    """
    if device is None:
        device = get_device()

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    checkpoint = torch.load(filename, map_location=device)

    # Support loading state_dict directly or from a checkpoint dict
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
        best_score = checkpoint.get("best_score", 0.0)
    else:
        model.load_state_dict(checkpoint)
        best_score = 0.0

    return model, best_score


def print_metrics(epoch, train_loss, val_loss, val_score):
    """
    Prints training and validation metrics with full precision.
    """
    print(
        f"Epoch {epoch}: Train Loss = {train_loss}, Val Loss = {val_loss}, Val AUC = {val_score}"
    )
