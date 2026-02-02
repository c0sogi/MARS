import os
import sys
import random
import shutil
import logging
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
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
    Used for tracking loss and accuracy during training.
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


def get_logger(log_file):
    """
    Creates a logger that writes to both a file and the console.
    """
    # Ensure the directory for the log file exists
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logging if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Define format
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def save_checkpoint(state, is_best, checkpoint_dir, fold_idx=None):
    """
    Saves the model state to a checkpoint file.
    If is_best is True, also copies the file to a 'best_model' file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    if fold_idx is not None:
        filename = f"checkpoint_fold_{fold_idx}.pth"
        best_filename = f"best_model_fold_{fold_idx}.pth"
    else:
        filename = "checkpoint.pth"
        best_filename = "best_model.pth"

    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_filepath = os.path.join(checkpoint_dir, best_filename)
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(model, filepath, device=None):
    """
    Loads model weights from a checkpoint file.
    Can handle checkpoints that are just the state_dict or a dict containing 'state_dict'.
    """
    if device is None:
        device = Config.DEVICE

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    # Load checkpoint
    checkpoint = torch.load(filepath, map_location=device)

    # Load state dict into model
    # Check if the checkpoint is a dictionary containing 'state_dict' or 'model_state_dict'
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Assume the dictionary itself is the state dict
            model.load_state_dict(checkpoint)
    else:
        # Fallback if checkpoint is not a dict (unlikely for standard saves but possible)
        model.load_state_dict(checkpoint)

    return model, checkpoint
