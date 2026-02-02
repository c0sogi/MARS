import os
import random
import numpy as np
import torch
import logging
import shutil
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(filename: str = "train.log"):
    """
    Initializes and returns a logger that outputs to both console and a file.
    The log file is saved in the working directory specified in Config.
    """
    log_file_path = os.path.join(Config.working_dir, filename)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create handlers
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file_path)

    # Create formatters and add it to handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    # Avoid adding multiple handlers if logger is already configured
    if not logger.handlers:
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)

    return logger


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
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


def save_checkpoint(state: dict, is_best: bool, filename: str = "checkpoint.pth"):
    """
    Saves the training checkpoint.

    Args:
        state (dict): The state dictionary containing model parameters, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): Name of the checkpoint file.
    """
    # Ensure the working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    filepath = os.path.join(Config.working_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_filepath = os.path.join(Config.working_dir, "best_model.pth")
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(model, filename: str, optimizer=None, scheduler=None, device=None):
    """
    Loads a checkpoint into the model (and optionally optimizer and scheduler).

    Args:
        model (torch.nn.Module): The model to load weights into.
        filename (str): The filename of the checkpoint to load (relative to working_dir or absolute).
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (torch.optim.lr_scheduler, optional): The scheduler to load state into.
        device (torch.device, optional): The device to map the checkpoint to.

    Returns:
        dict: The loaded checkpoint dictionary (useful for retrieving epoch or best_score).
    """
    if device is None:
        device = Config.device

    # Check if filename exists as provided, otherwise assume it's in working_dir
    if os.path.exists(filename):
        filepath = filename
    elif not os.path.isabs(filename):
        filepath = os.path.join(Config.working_dir, filename)
    else:
        filepath = filename

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    print(f"Loading checkpoint from {filepath}...")
    checkpoint = torch.load(filepath, map_location=device)

    # Load model state
    # Handle DataParallel wrapping if necessary (remove 'module.' prefix)
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    # Load optimizer state
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Load scheduler state
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    print(
        f"Loaded checkpoint '{filepath}' (epoch {checkpoint.get('epoch', 'Unknown')})"
    )

    return checkpoint
