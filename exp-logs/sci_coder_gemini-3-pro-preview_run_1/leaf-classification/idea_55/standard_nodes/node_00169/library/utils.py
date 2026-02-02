import os
import sys
import random
import logging
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from library import config


def set_seed(seed=config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(name="leaf_classifier", log_file=None, level=logging.INFO):
    """
    Configures and returns a logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def enforce_float64(data):
    """
    Ensures the input data (DataFrame or array) is in float64 precision.
    """
    if isinstance(data, pd.DataFrame):
        # Convert all float columns to float64
        float_cols = data.select_dtypes(include=["float16", "float32"]).columns
        if len(float_cols) > 0:
            data[float_cols] = data[float_cols].astype(np.float64)
        # Ensure numpy arrays inside are float64 if ambiguous
        return data
    elif isinstance(data, np.ndarray):
        return data.astype(np.float64)
    return data


def compute_log_loss(y_true, y_pred, labels):
    """
    Computes the multi-class log loss according to the competition metric.

    Logic:
    1. Normalize rows (predictions rescaled to sum to 1).
    2. Clip probabilities to [1e-15, 1-1e-15].
    3. Compute sklearn log_loss.

    Args:
        y_true: Array-like of ground truth labels (strings or ints).
        y_pred: Array-like of predicted probabilities (shape: n_samples x n_classes).
        labels: List of class labels corresponding to columns of y_pred.

    Returns:
        float: The log loss value.
    """
    # Ensure float64 for precision during metric calc
    y_pred = np.array(y_pred, dtype=np.float64)

    # 1. Rescale: each row divided by row sum
    row_sums = y_pred.sum(axis=1)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1.0
    y_pred_norm = y_pred / row_sums[:, np.newaxis]

    # 2. Clip: max(min(p, 1-10^-15), 10^-15)
    eps = 1e-15
    y_pred_clipped = np.clip(y_pred_norm, eps, 1 - eps)

    # 3. Calculate Log Loss
    # labels argument ensures correct mapping if y_true are strings
    return log_loss(y_true, y_pred_clipped, labels=labels)


def save_submission(ids, probas, classes, output_path):
    """
    Saves the predictions to a CSV file in the required format.

    Format:
    id,Species1,Species2,...
    2,0.1,0.5,...

    Args:
        ids: Array-like of image IDs.
        probas: Array-like of probabilities (n_samples x n_classes).
        classes: List of class names (column headers).
        output_path: Path to save the CSV.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame
    submission_df = pd.DataFrame(probas, columns=classes)
    submission_df.insert(0, "id", ids)

    # Save
    submission_df.to_csv(output_path, index=False)


def load_csv_data(path):
    """
    Simple wrapper to load CSV data.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)
