import os
import sys
import random
import shutil
import logging
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    Ensures deterministic behavior for CuDNN backends.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Enforce deterministic algorithms to ensure reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_file, name="root"):
    """
    Configures a logger to write messages to both a file and the console (stdout).

    Args:
        log_file (str): Path to the log file.
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Remove existing handlers to prevent duplication if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler
    # Ensure the directory for the log file exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    return logger


def save_checkpoint(state, is_best, fold):
    """
    Saves the model state to a checkpoint file. If the model is the best so far,
    copies it to a separate 'best' file.

    Args:
        state (dict): State dictionary (model weights, optimizer, epoch, etc.).
        is_best (bool): Flag indicating if this is the best model so far.
        fold (int): Current fold index (0-based) to distinguish files.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    filename = os.path.join(Config.WORKING_DIR, f"checkpoint_fold_{fold}.pth")
    torch.save(state, filename)

    if is_best:
        best_filename = os.path.join(Config.WORKING_DIR, f"model_best_fold_{fold}.pth")
        shutil.copyfile(filename, best_filename)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Loads a checkpoint into the model and optionally the optimizer.

    Args:
        checkpoint_path (str): Path to the .pth file.
        model (torch.nn.Module): The model instance to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.

    Returns:
        dict: The loaded checkpoint dictionary.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Load to CPU first to avoid OOM or device mismatch, then let model.load_state_dict handle device placement
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking losses and metrics during training loops.
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
