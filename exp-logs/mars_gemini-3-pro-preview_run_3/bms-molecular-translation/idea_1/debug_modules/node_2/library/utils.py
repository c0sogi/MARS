import os
import random
import shutil
import numpy as np
import torch
import nltk
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to 42.
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


def compute_levenshtein(predictions, targets):
    """
    Computes the mean Levenshtein distance between predictions and targets.

    Args:
        predictions (list of str): List of predicted InChI strings.
        targets (list of str): List of ground truth InChI strings.

    Returns:
        float: Mean Levenshtein distance.
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"Predictions ({len(predictions)}) and targets ({len(targets)}) must have the same length."
        )

    total_distance = 0
    count = len(predictions)

    if count == 0:
        return 0.0

    for pred, target in zip(predictions, targets):
        # nltk.edit_distance calculates the Levenshtein distance
        dist = nltk.edit_distance(pred, target)
        total_distance += dist

    return total_distance / count


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model parameters, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): Filename for the checkpoint.
    """
    # Ensure checkpoint directory exists (though Config creates it)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    filepath = os.path.join(Config.CHECKPOINT_DIR, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(Config.CHECKPOINT_DIR, "model_best.pth.tar")
        shutil.copyfile(filepath, best_path)


def load_checkpoint(
    filename, model, optimizer=None, scheduler=None, device=Config.DEVICE
):
    """
    Loads a model checkpoint.

    Args:
        filename (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (torch.optim.lr_scheduler, optional): The scheduler to load state into.
        device (torch.device): Device to map the location to.

    Returns:
        tuple: (start_epoch, best_metric)
            start_epoch (int): The epoch to resume from.
            best_metric (float): The best metric value (e.g., min loss or distance) if present.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    checkpoint = torch.load(filename, map_location=device)

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    start_epoch = checkpoint.get("epoch", 0)
    best_metric = checkpoint.get("best_metric", float("inf"))

    print(f"Loaded checkpoint '{filename}' (epoch {start_epoch})")
    return start_epoch, best_metric
