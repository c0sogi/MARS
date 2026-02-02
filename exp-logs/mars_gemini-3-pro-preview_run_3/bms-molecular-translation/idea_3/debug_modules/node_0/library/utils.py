import os
import random
import shutil
import numpy as np
import torch
import nltk
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


def seed_everything(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_levenshtein(predictions, targets):
    """
    Computes the mean Levenshtein distance between predictions and targets.

    Args:
        predictions (list of str): Predicted InChI strings.
        targets (list of str): Ground truth InChI strings.

    Returns:
        float: Mean Levenshtein distance.
    """
    if not predictions or not targets:
        return 0.0

    total_distance = 0
    # Calculate edit distance for each pair
    for p, t in zip(predictions, targets):
        total_distance += nltk.edit_distance(p, t)

    return total_distance / len(predictions)


def save_checkpoint(state, is_best, filename=Config.CHECKPOINT_PATH):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model parameters, optimizer, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): Path to save the checkpoint.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, Config.BEST_MODEL_PATH)


def load_checkpoint(
    model, filename=Config.BEST_MODEL_PATH, optimizer=None, scheduler=None
):
    """
    Loads a model checkpoint.

    Args:
        model (torch.nn.Module): The model to load weights into.
        filename (str): Path to the checkpoint file.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (torch.optim.lr_scheduler, optional): Scheduler to load state into.

    Returns:
        tuple: (start_epoch, best_score)
    """
    if not os.path.exists(filename):
        print(f"Checkpoint file not found at {filename}")
        return 0, float("inf")

    checkpoint = torch.load(filename, map_location="cpu")

    # Handle state dict keys (e.g., remove 'module.' prefix if saved from DataParallel)
    state_dict = checkpoint["state_dict"]
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    start_epoch = checkpoint.get("epoch", 0)
    best_score = checkpoint.get("best_score", float("inf"))

    print(
        f"Loaded checkpoint '{filename}' (epoch {start_epoch}, score {best_score:.4f})"
    )
    return start_epoch, best_score
