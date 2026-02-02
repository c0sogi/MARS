import os
import shutil
import torch
import nltk
from library.config import Config


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class LevenshteinMetric:
    """
    Computes the Mean Levenshtein distance between predicted and ground truth strings.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_distance = 0.0
        self.count = 0

    def update(self, preds, targets):
        """
        Update the metric with a batch of predictions and targets.

        Args:
            preds (list of str): Predicted InChI strings.
            targets (list of str): Ground truth InChI strings.
        """
        for p, t in zip(preds, targets):
            # Calculate Levenshtein distance using NLTK
            # edit_distance computes the number of operations to transform p to t
            dist = nltk.edit_distance(p, t)
            self.total_distance += dist
            self.count += 1

    def get_avg_score(self):
        """Returns the average Levenshtein distance."""
        if self.count == 0:
            return 0.0
        return self.total_distance / self.count


def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): Filename for the checkpoint.
    """
    # Ensure the working directory exists (redundant with Config but safe)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    filepath = os.path.join(Config.WORKING_DIR, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(Config.WORKING_DIR, "model_best.pth")
        shutil.copyfile(filepath, best_path)


def load_checkpoint(model, optimizer=None, scheduler=None, filename="checkpoint.pth"):
    """
    Loads a checkpoint into the model and optional optimizer/scheduler.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (torch.optim.lr_scheduler, optional): Scheduler to load state into.
        filename (str): Filename of the checkpoint to load.

    Returns:
        start_epoch (int): The epoch to resume from (0 if no checkpoint found).
        best_score (float): The best score recorded (inf if no checkpoint found).
    """
    filepath = os.path.join(Config.WORKING_DIR, filename)
    if not os.path.exists(filepath):
        # Return default values if no checkpoint exists
        return 0, float("inf")

    print(f"Loading checkpoint '{filepath}'")
    checkpoint = torch.load(filepath, map_location=Config.DEVICE)

    # Load model state
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)  # Fallback if only state_dict was saved

    # Load optimizer state
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Load scheduler state
    if scheduler and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    start_epoch = checkpoint.get("epoch", 0)
    # Default to inf because lower Levenshtein distance is better
    best_score = checkpoint.get("best_score", float("inf"))

    print(
        f"Loaded checkpoint '{filepath}' (epoch {start_epoch}, best_score {best_score})"
    )
    return start_epoch, best_score
