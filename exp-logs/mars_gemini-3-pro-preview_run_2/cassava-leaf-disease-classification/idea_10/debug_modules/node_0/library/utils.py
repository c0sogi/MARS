import os
import sys
import random
import logging
import shutil
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED) -> None:
    """
    Seeds all random number generators to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(filename: str = "train.log") -> logging.Logger:
    """
    Creates and configures a logger that writes to a file and stdout.

    Args:
        filename (str): The name of the log file. Defaults to "train.log".

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Ensure the working directory exists
    Config.setup_directories()
    log_file_path = os.path.join(Config.WORKING_DIR, filename)

    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)

    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        # File Handler
        fh = logging.FileHandler(log_file_path)
        fh.setLevel(logging.INFO)

        # Stream Handler (stdout)
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        sh.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(sh)

    return logger


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

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state: dict, is_best: bool, fold: int) -> None:
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        fold (int): The current fold number.
    """
    Config.setup_directories()

    filename = f"checkpoint_fold_{fold}.pth"
    filepath = os.path.join(Config.WORKING_DIR, filename)

    torch.save(state, filepath)

    if is_best:
        best_filename = f"best_model_fold_{fold}.pth"
        best_filepath = os.path.join(Config.WORKING_DIR, best_filename)
        shutil.copyfile(filepath, best_filepath)
