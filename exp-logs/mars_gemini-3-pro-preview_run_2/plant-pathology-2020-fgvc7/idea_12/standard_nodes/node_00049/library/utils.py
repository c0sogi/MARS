import os
import random
import logging
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for python, numpy, and torch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str = "train"):
    """
    Creates and configures a logger.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def worker_init_fn(worker_id: int):
    """
    Worker initialization function for DataLoader to ensure deterministic
    data augmentation and processing in multi-process data loading.

    Args:
        worker_id (int): The ID of the worker process.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def calculate_pos_weights(metadata_df: pd.DataFrame) -> torch.Tensor:
    """
    Calculates inverse class frequency weights for BCEWithLogitsLoss.

    Strategy:
    - Target 0 (Is_Rust): Positive if label is 'rust' OR 'multiple_diseases'.
    - Target 1 (Is_Scab): Positive if label is 'scab' OR 'multiple_diseases'.

    Formula: pos_weight = number_of_negatives / number_of_positives

    Args:
        metadata_df (pd.DataFrame): The training metadata containing one-hot labels.

    Returns:
        torch.Tensor: A tensor of shape [2] containing weights for Rust and Scab.
    """
    # Ensure we are working with the correct columns
    # The dataframe is expected to have 'rust', 'scab', 'multiple_diseases'

    # 1. Construct Binary Targets
    # Rust is present in 'rust' class and 'multiple_diseases' class
    is_rust = (metadata_df["rust"] == 1) | (metadata_df["multiple_diseases"] == 1)

    # Scab is present in 'scab' class and 'multiple_diseases' class
    is_scab = (metadata_df["scab"] == 1) | (metadata_df["multiple_diseases"] == 1)

    # 2. Calculate Counts
    num_pos_rust = is_rust.sum()
    num_neg_rust = len(metadata_df) - num_pos_rust

    num_pos_scab = is_scab.sum()
    num_neg_scab = len(metadata_df) - num_pos_scab

    # 3. Calculate Weights (Neg/Pos)
    # Add epsilon to avoid division by zero if a class is missing (unlikely in this dataset)
    eps = 1e-6
    weight_rust = num_neg_rust / (num_pos_rust + eps)
    weight_scab = num_neg_scab / (num_pos_scab + eps)

    weights = torch.tensor([weight_rust, weight_scab], dtype=torch.float32)

    return weights
