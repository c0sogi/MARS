import os
import sys
import random
import logging
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed: int = Config.SEED):
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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(
    name: str = "leaf_classification", log_file: str = None, level: int = logging.INFO
):
    """
    Configures and returns a logger that writes to stdout and optionally a file.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file. If provided, logs will be written here.
        level (int): Logging level (e.g., logging.INFO).

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to prevent duplicate logging if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def load_metadata(
    split: str = "train",
    debug: bool = Config.DEBUG,
    sample_size: int = Config.DEBUG_SAMPLE_SIZE,
):
    """
    Loads the metadata CSV for the requested split.

    Args:
        split (str): One of 'train', 'val', or 'test'.
        debug (bool): If True, returns only the first `sample_size` rows.
        sample_size (int): Number of rows to return in debug mode.

    Returns:
        pd.DataFrame: The loaded metadata DataFrame.
    """
    if split == "train":
        path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        path = Config.VAL_METADATA_PATH
    elif split == "test":
        path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(
            f"Invalid split '{split}'. Must be one of: 'train', 'val', 'test'."
        )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at: {path}")

    df = pd.read_csv(path)

    if debug:
        df = df.head(sample_size)

    return df
