import os
import sys
import shutil
import logging
import torch
from library.config import Config, set_seed


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training loops.
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
        return f"{self.name} {self.val}{self.fmt} ({self.avg}{self.fmt})"


def get_logger(log_file):
    """
    Creates and configures a logger that writes to both a file and the console.

    Args:
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name=log_file)
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times if get_logger is called repeatedly
    if not logger.handlers:
        # File Handler
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Console Handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


def save_checkpoint(state, is_best, checkpoint_dir, filename="checkpoint.pth"):
    """
    Saves the model state to a checkpoint file.

    Args:
        state (dict): State dictionary containing model, optimizer, epoch, etc.
        is_best (bool): If True, copies the checkpoint to 'model_best.pth'.
        checkpoint_dir (str): Directory where checkpoints are saved.
        filename (str): Name of the checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "model_best.pth")
        shutil.copyfile(filepath, best_path)


def load_checkpoint(
    filepath, model, optimizer=None, scheduler=None, device=Config.DEVICE
):
    """
    Loads a checkpoint into the model, and optionally optimizer and scheduler.

    Args:
        filepath (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler to load state into.
        device (torch.device): Device to map the checkpoint to.

    Returns:
        dict: The loaded checkpoint dictionary.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No checkpoint found at '{filepath}'")

    # Cite {debug_lesson_8}: Explicitly set weights_only=False to allow loading checkpoints with numpy scalars
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    # Load model state dict
    # Handle potential DataParallel wrapping (keys starting with 'module.')
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k

        # Cite {debug_lesson_16}: Strip SWA-specific keys (n_averaged) to prevent unexpected key errors
        if name == "n_averaged":
            continue

        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint
