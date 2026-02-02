import os
import sys
import random
import logging
import shutil
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(log_file):
    """
    Creates and configures a logger that outputs to both a file and standard output.

    Args:
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("CassavaLogger")
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times if get_logger is called repeatedly
    if not logger.handlers:
        # Create formatters
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Stream Handler (Console)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # File Handler
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking metrics like loss and accuracy.
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


def save_checkpoint(state, is_best, checkpoint_dir=None, fold=0):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        checkpoint_dir (str, optional): Directory to save the checkpoint.
                                        Defaults to Config.working_dir if not provided.
        fold (int): The current fold number.
    """
    if checkpoint_dir is None:
        checkpoint_dir = Config.working_dir

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save the latest checkpoint
    filename = os.path.join(checkpoint_dir, f"checkpoint_fold_{fold}.pth")
    torch.save(state, filename)

    # If it is the best model, save a copy
    if is_best:
        best_filename = os.path.join(checkpoint_dir, f"best_model_fold_{fold}.pth")
        shutil.copyfile(filename, best_filename)
