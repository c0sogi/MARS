import os
import random
import shutil
import logging
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN to guarantee reproducibility
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
    # Create logger
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)

    # Check if handlers already exist to avoid duplicates if function is called multiple times
    if not logger.handlers:
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter and add to handlers
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def save_checkpoint(state, is_best, checkpoint_dir, fold):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model parameters, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        checkpoint_dir (str): Directory to save the checkpoint.
        fold (int): Current fold number.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save the current state (latest checkpoint)
    filename = os.path.join(checkpoint_dir, f"checkpoint_fold_{fold}.pth")
    torch.save(state, filename)

    # If this is the best model, create a copy
    if is_best:
        best_filename = os.path.join(checkpoint_dir, f"model_best_fold_{fold}.pth")
        shutil.copyfile(filename, best_filename)


def load_checkpoint(checkpoint_path, model, optimizer=None, device="cpu"):
    """
    Loads a checkpoint into the model and optionally the optimizer.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str): Device to map the storage to ('cpu' or 'cuda').

    Returns:
        dict: The loaded checkpoint dictionary, allowing access to other saved metadata (e.g. epoch).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Load checkpoint with appropriate device mapping
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model weights
    # strict=False allows loading if there are minor mismatches, but usually strict=True is safer.
    # We use strict=True assuming the architecture matches exactly.
    model.load_state_dict(checkpoint["state_dict"])

    # Load optimizer state if provided and present in checkpoint
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint
