import os
import sys
import random
import logging
import shutil
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name, log_file=None):
    """
    Creates and configures a logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Stream Handler (stdout)
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


class AverageMeter:
    """
    Computes and stores the average and current value.
    """

    def __init__(self, name="Meter"):
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


def save_checkpoint(
    state,
    is_best,
    checkpoint_dir=Config.WORKING_DIR,
    filename="checkpoint.pth",
    best_filename="best_model.pth",
):
    """
    Saves the model checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    file_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, file_path)

    if is_best:
        best_path = os.path.join(checkpoint_dir, best_filename)
        shutil.copyfile(file_path, best_path)


def load_checkpoint(filepath, model, optimizer=None, device=Config.DEVICE):
    """
    Loads a model checkpoint.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found at {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    # Support loading both full checkpoint dicts and direct state dicts
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Handle DataParallel wrapping if necessary (remove 'module.' prefix)
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint


def save_numpy_cache(data, filepath):
    """
    Saves data to a .npy file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, data)


def load_numpy_cache(filepath):
    """
    Loads data from a .npy file if it exists.
    """
    if os.path.exists(filepath):
        return np.load(filepath, allow_pickle=True)
    return None


def save_parquet_cache(df, filepath):
    """
    Saves a pandas DataFrame to a parquet file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_parquet(filepath, index=False)


def load_parquet_cache(filepath):
    """
    Loads a pandas DataFrame from a parquet file if it exists.
    """
    if os.path.exists(filepath):
        return pd.read_parquet(filepath)
    return None


def log_metrics(logger, metrics, prefix=""):
    """
    Logs a dictionary of metrics with full precision.
    """
    msg_parts = []
    for k, v in metrics.items():
        msg_parts.append(f"{k}: {v}")

    msg = ", ".join(msg_parts)
    if prefix:
        logger.info(f"{prefix} {msg}")
    else:
        logger.info(msg)
