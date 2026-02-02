import os
import random
import shutil
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Enforce deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Returns the appropriate torch device based on availability.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_fn(batch):
    """
    Custom collate function for object detection.

    Args:
        batch: List of tuples (image, target)

    Returns:
        Tuple of (tuple(images), tuple(targets))
    """
    return tuple(zip(*batch))


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
    """

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
        # Print full precision as requested
        return f"{self.name} {self.val} ({self.avg})"


def save_checkpoint(
    state, is_best, checkpoint_dir=Config.WORKING_DIR, filename="checkpoint.pth"
):
    """
    Saves the model checkpoint.

    Args:
        state (dict): The model state dictionary (params, optimizer, epoch, etc.).
        is_best (bool): Whether this checkpoint represents the best model so far.
        checkpoint_dir (str): Directory to save the checkpoint.
        filename (str): Filename for the checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_filepath = os.path.join(checkpoint_dir, "best_model.pth")
        shutil.copyfile(filepath, best_filepath)
        print(f"Saved new best model to {best_filepath}")


def log_metrics(metrics_dict, prefix=""):
    """
    Prints metrics with full precision.

    Args:
        metrics_dict (dict): Dictionary of metric names and values.
        prefix (str): Optional prefix for the log message.
    """
    log_str = prefix
    for key, value in metrics_dict.items():
        log_str += f" {key}: {value}"
    print(log_str)
