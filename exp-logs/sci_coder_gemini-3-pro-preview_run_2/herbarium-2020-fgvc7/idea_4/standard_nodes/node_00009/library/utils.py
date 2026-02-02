import os
import random
import shutil
import logging
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logger(log_file):
    """
    Creates and configures a logger that writes to both a file and the console.

    Args:
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("experiment_logger")
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logging if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Ensure directory exists for the log file
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # File Handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def save_checkpoint(state, is_best, working_dir, filename="checkpoint.pth"):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model parameters, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        working_dir (str): Directory to save the checkpoint.
        filename (str): Name of the checkpoint file.
    """
    os.makedirs(working_dir, exist_ok=True)
    checkpoint_path = os.path.join(working_dir, filename)
    torch.save(state, checkpoint_path)

    if is_best:
        best_path = os.path.join(working_dir, "best_model.pth")
        shutil.copyfile(checkpoint_path, best_path)


def load_checkpoint(
    checkpoint_path, model, optimizer=None, scheduler=None, device="cpu"
):
    """
    Loads a model checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The scheduler to load state into.
        device (str or torch.device): Device to map the location to.

    Returns:
        dict: The loaded checkpoint dictionary.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    # Handle cases where DataParallel was used (keys start with module.)
    state_dict = checkpoint["state_dict"]
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint
