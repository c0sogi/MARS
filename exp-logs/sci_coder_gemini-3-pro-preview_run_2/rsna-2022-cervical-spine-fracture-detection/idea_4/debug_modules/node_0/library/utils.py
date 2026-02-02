import os
import sys
import logging
import torch
import shutil
from library.config import Config, seed_everything


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking metrics like loss and accuracy during training.
    """

    def __init__(self, name="Metric"):
        self.name = name
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
        return f"{self.name}: {self.val} (Avg: {self.avg})"


def get_logger(name=Config.PROJECT_NAME, log_file=None):
    """
    Configures and returns a logger instance.
    Writes to sys.stdout and optionally to a file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Remove existing handlers to prevent duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # Stream Handler (Console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def save_checkpoint(state, is_best, filepath=Config.CHECKPOINT_PATH):
    """
    Saves the model state to a file.

    Args:
        state (dict): The state dictionary containing model parameters, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filepath (str): Path to save the checkpoint.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # In this workflow, we generally save the checkpoint.
    # If it is the best model, it is saved directly to the target path.
    # We can also implement logic to keep a separate 'last.pth' if needed,
    # but here we prioritize the requirements.
    if is_best:
        torch.save(state, filepath)


def load_checkpoint(
    model, filepath=Config.CHECKPOINT_PATH, optimizer=None, scheduler=None
):
    """
    Loads a model checkpoint from the specified file.

    Args:
        model (torch.nn.Module): The model to load weights into.
        filepath (str): Path to the checkpoint file.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (torch.optim.lr_scheduler, optional): Scheduler to load state into.

    Returns:
        dict: The loaded checkpoint dictionary, or None if file not found.
    """
    if not os.path.exists(filepath):
        # We log this but do not raise error to allow starting from scratch
        print(f"No checkpoint found at {filepath}. Starting from scratch.")
        return None

    print(f"Loading checkpoint from {filepath}...")
    checkpoint = torch.load(filepath, map_location=Config.DEVICE)

    # Handle state dict loading (handling potential DataParallel prefix)
    state_dict = checkpoint["state_dict"]

    # If model is not DataParallel but saved state has 'module.', remove it
    if not isinstance(model, torch.nn.DataParallel):
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    print(f"Checkpoint loaded successfully.")
    return checkpoint


def print_metrics(metrics, logger=None, prefix=""):
    """
    Prints validation metrics with full precision.

    Args:
        metrics (dict): Dictionary of metric names and values.
        logger (logging.Logger, optional): Logger to use. If None, uses print.
        prefix (str): Optional prefix for the log message.
    """
    # Format string to ensure full precision (no rounding)
    metric_strs = [f"{k}: {v}" for k, v in metrics.items()]
    message = f"{prefix} " + " | ".join(metric_strs)

    if logger:
        logger.info(message)
    else:
        print(message)
