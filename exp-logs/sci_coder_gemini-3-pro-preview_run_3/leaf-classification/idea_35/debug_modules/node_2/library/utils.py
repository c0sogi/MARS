import os
import random
import logging
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str = "leaf_classification", log_file: str = None):
    """
    Configures and returns a logger instance.

    Args:
        name (str): Name of the logger.
        log_file (str): Path to the log file. If None, uses execution.log in WORKING_DIR.

    Returns:
        logging.Logger: Configured logger.
    """
    if log_file is None:
        log_file = os.path.join(Config.WORKING_DIR, "execution.log")

    # Ensure directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def clip_and_normalize(probs: np.ndarray) -> np.ndarray:
    """
    Applies the competition-specific normalization and clipping.
    1. Rescale: Divide each row by the row sum.
    2. Clip: Replace probabilities with max(min(p, 1-10^-15), 10^-15).

    Args:
        probs (np.ndarray): Raw probability matrix (N_samples, N_classes).

    Returns:
        np.ndarray: Processed probability matrix.
    """
    # 1. Rescale (Normalize rows to sum to 1)
    # Add epsilon to avoid division by zero if a row is all zeros (unlikely but safe)
    row_sums = probs.sum(axis=1, keepdims=True)
    probs_norm = probs / (row_sums + 1e-15)

    # 2. Clip to avoid log function extremes
    # Range: [1e-15, 1 - 1e-15]
    epsilon = 1e-15
    probs_clipped = np.clip(probs_norm, epsilon, 1 - epsilon)

    return probs_clipped


def calculate_log_loss(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list = None
) -> float:
    """
    Calculates the multi-class log loss metric.

    Args:
        y_true (np.ndarray): True class indices or one-hot encoded labels.
        y_pred (np.ndarray): Predicted probabilities.
        labels (list): List of class labels (optional, for sklearn).

    Returns:
        float: The calculated log loss.
    """
    # Apply competition specific post-processing
    y_pred_processed = clip_and_normalize(y_pred)

    # Calculate log loss
    # If y_true are indices, sklearn handles it. If one-hot, sklearn handles it.
    score = log_loss(y_true, y_pred_processed, labels=labels)

    # Print full precision as requested
    print(f"Validation Log Loss: {score}")

    return score


def save_submission(
    ids: np.ndarray,
    classes: list,
    probs: np.ndarray,
    filename: str = Config.SUBMISSION_FILE_PATH,
):
    """
    Saves the predictions to a CSV file in the required format.

    Format:
    id,Acer_Capillipes,Acer_Circinatum,...
    2,0.1,0.5,...

    Args:
        ids (np.ndarray): Array of image IDs.
        classes (list): List of species names (column headers).
        probs (np.ndarray): Matrix of predicted probabilities (N_samples, N_classes).
        filename (str): Output file path.
    """
    # Ensure probabilities are normalized and clipped before saving
    # While the evaluation metric does this, submitting clean probabilities is good practice.
    # Note: The prompt says "submitted probabilities... are rescaled prior to being scored",
    # so we can submit raw probabilities, but normalized ones are safer.
    probs_processed = clip_and_normalize(probs)

    # Create DataFrame
    df = pd.DataFrame(probs_processed, columns=classes)

    # Insert ID column at the beginning
    df.insert(0, "id", ids.astype(int))

    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Submission saved to {filename}")


def load_data_from_cache(cache_path: str):
    """
    Helper to load numpy data from cache if it exists.

    Args:
        cache_path (str): Path to the .npy file.

    Returns:
        np.ndarray or None: Loaded data or None if not found.
    """
    if os.path.exists(cache_path):
        try:
            return np.load(cache_path, allow_pickle=True)
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}")
            return None
    return None


def save_data_to_cache(data: np.ndarray, cache_path: str):
    """
    Helper to save numpy data to cache.

    Args:
        data (np.ndarray): Data to save.
        cache_path (str): Path to the .npy file.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, data)
