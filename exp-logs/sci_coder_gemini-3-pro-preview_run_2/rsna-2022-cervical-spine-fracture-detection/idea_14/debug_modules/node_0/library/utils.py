import os
import sys
import random
import time
import logging
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Ensures deterministic behavior for CUDA operations.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # deterministic algorithms for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Returns the appropriate PyTorch device (CUDA or CPU) based on availability
    and configuration.
    """
    return torch.device(Config.DEVICE)


def setup_logger(log_file: str = None) -> logging.Logger:
    """
    Sets up a logger that writes to both console and a file.
    If log_file is not provided, it defaults to 'train.log' in the Config.OUTPUT_DIR.
    """
    if log_file is None:
        # Ensure output directory exists
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        log_file = os.path.join(Config.OUTPUT_DIR, "train.log")

    # Create directory for the log file if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("RSNA_Cervical_Spine")
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if setup_logger is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter
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


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training loops.
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


def format_time(seconds: float) -> str:
    """
    Converts seconds to a formatted string (HH:MM:SS).
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"


def save_checkpoint(model, optimizer, epoch, scheduler=None, filename="checkpoint.pth"):
    """
    Saves the model checkpoint including optimizer and scheduler states.
    """
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(Config.OUTPUT_DIR, filename)

    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()

    torch.save(state, checkpoint_path)


def print_config(logger: logging.Logger = None):
    """
    Prints the configuration settings to the provided logger or stdout.
    """
    config_dict = {
        k: v
        for k, v in Config.__dict__.items()
        if not k.startswith("__") and not callable(v)
    }
    msg = "Configuration:\n"
    for k, v in config_dict.items():
        msg += f"  {k}: {v}\n"

    if logger:
        logger.info(msg)
    else:
        print(msg)
