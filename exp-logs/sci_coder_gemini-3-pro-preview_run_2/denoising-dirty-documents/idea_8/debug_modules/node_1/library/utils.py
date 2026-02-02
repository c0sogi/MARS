import os
import random
import shutil
import logging
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=42):
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
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_rmse(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error (RMSE) between true and predicted values.
    Handles both NumPy arrays and PyTorch tensors.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth values.
        y_pred (np.ndarray or torch.Tensor): Predicted values.

    Returns:
        float: The RMSE value.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are float for accurate calculation
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)

    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def get_logger(name, log_file=None):
    """
    Creates and returns a logger instance.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def save_checkpoint(state, is_best, checkpoint_dir, filename="checkpoint.pth"):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model, optimizer, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        checkpoint_dir (str): Directory to save the checkpoint.
        filename (str): Name of the checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = Config.MODEL_PATH
        # Ensure the directory for the best model exists
        os.makedirs(os.path.dirname(best_path), exist_ok=True)
        shutil.copyfile(filepath, best_path)


def load_checkpoint(
    checkpoint_path, model, optimizer=None, scheduler=None, device="cpu"
):
    """
    Loads a checkpoint into the model, optimizer, and scheduler.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler to load state into.
        device (str or torch.device): Device to map the checkpoint to.

    Returns:
        tuple: (start_epoch, best_metric)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    start_epoch = checkpoint.get("epoch", 0)
    best_metric = checkpoint.get("best_metric", float("inf"))

    return start_epoch, best_metric
