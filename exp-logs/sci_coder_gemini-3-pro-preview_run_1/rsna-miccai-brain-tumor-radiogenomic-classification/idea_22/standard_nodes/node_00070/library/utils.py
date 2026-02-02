import os
import random
import shutil
import numpy as np
import torch
from library import config


def seed_everything(seed=config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def save_checkpoint(state, is_best, fold_idx=None, filename="checkpoint.pth"):
    """
    Saves the model checkpoint.

    Args:
        state (dict): The state dictionary containing model weights, optimizer state, etc.
        is_best (bool): If True, copies the checkpoint to a 'best_model' file.
        fold_idx (int, optional): The current fold index for cross-validation naming.
        filename (str): Base filename for the checkpoint.
    """
    if fold_idx is not None:
        name, ext = os.path.splitext(filename)
        filename = f"{name}_fold{fold_idx}{ext}"

    filepath = os.path.join(config.WORKING_DIR, filename)
    torch.save(state, filepath)

    if is_best:
        best_filename = "best_model.pth"
        if fold_idx is not None:
            best_filename = f"best_model_fold{fold_idx}.pth"

        best_filepath = os.path.join(config.WORKING_DIR, best_filename)
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(model, filepath, device=config.DEVICE):
    """
    Loads model weights from a checkpoint file.

    Args:
        model (torch.nn.Module): The model to load weights into.
        filepath (str): Path to the checkpoint file.
        device (str): Device to map the location to ('cpu' or 'cuda').

    Returns:
        dict: The loaded checkpoint dictionary, or None if file not found.
    """
    if not os.path.exists(filepath):
        print(f"Checkpoint file not found at: {filepath}")
        return None

    checkpoint = torch.load(filepath, map_location=device)

    # Support loading both full checkpoint dicts and direct state dicts
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return checkpoint


def print_metrics(metrics, prefix=""):
    """
    Prints metric values with full precision.

    Args:
        metrics (dict): Dictionary of metric names and values.
        prefix (str): Optional prefix string for the log message.
    """
    log_parts = []
    if prefix:
        log_parts.append(f"[{prefix}]")

    for key, value in metrics.items():
        log_parts.append(f"{key}: {value}")

    print(" ".join(log_parts))
