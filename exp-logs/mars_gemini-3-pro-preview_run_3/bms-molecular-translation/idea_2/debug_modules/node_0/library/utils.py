import os
import shutil
import torch
import nltk
import numpy as np
from library.config import Config


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


def compute_levenshtein(predictions, targets):
    """
    Computes the mean Levenshtein distance between predictions and targets.

    Args:
        predictions (list of str): List of predicted InChI strings.
        targets (list of str): List of ground truth InChI strings.

    Returns:
        float: The mean Levenshtein distance.
    """
    scores = []
    for pred, target in zip(predictions, targets):
        # nltk.edit_distance calculates the Levenshtein distance
        score = nltk.edit_distance(pred, target)
        scores.append(score)

    return np.mean(scores)


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    """
    Saves the training checkpoint.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): Filename for the checkpoint.
    """
    # Ensure checkpoint directory exists
    checkpoint_dir = os.path.join(Config.WORKING_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "model_best.pth.tar")
        shutil.copyfile(filepath, best_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    Loads a checkpoint into the model and optionally optimizer and scheduler.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The scheduler to load state into.

    Returns:
        int: The epoch to resume from (if available in checkpoint), else 0.
        float: The best metric value (if available), else None.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Error: No checkpoint found at {checkpoint_path}")
        return 0, None

    print(f"Loading checkpoint '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    start_epoch = checkpoint.get("epoch", 0)
    best_metric = checkpoint.get("best_metric", None)

    print(f"Loaded checkpoint '{checkpoint_path}' (epoch {start_epoch})")
    return start_epoch, best_metric
