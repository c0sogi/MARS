import os
import sys
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from library.config import Config


def setup_logger(name="rsna_fracture", log_file=None, level=logging.INFO):
    """
    Sets up a logger that outputs to both console and a file.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file. If None, uses default from Config.
        level (int): Logging level (e.g., logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    if log_file is None:
        log_file = os.path.join(Config.LOG_DIR, "train.log")

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        # File Handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)

        # Console Handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add handlers
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


def window_dicom(image, center=Config.WINDOW_CENTER, width=Config.WINDOW_WIDTH):
    """
    Applies a windowing function to the DICOM pixel array (Hounsfield Units).
    Clips values to the window and normalizes to [0, 1].

    Args:
        image (np.ndarray): Raw pixel array (Hounsfield Units).
        center (float): Window center (level).
        width (float): Window width.

    Returns:
        np.ndarray: Windowed and normalized image in range [0, 1].
    """
    min_val = center - width / 2.0
    max_val = center + width / 2.0

    # Clip values to the window
    image = np.clip(image, min_val, max_val)

    # Normalize to [0, 1]
    if max_val != min_val:
        image = (image - min_val) / (max_val - min_val)
    else:
        image = image - min_val  # Should be 0s

    return image.astype(np.float32)


def calculate_weighted_log_loss(y_true, y_pred, eps=1e-15):
    """
    Calculates the weighted multi-label logarithmic loss for the competition.

    Weights:
        patient_overall: 1.0
        C1-C7: 1/7 (~0.1428)

    Args:
        y_true (np.ndarray or pd.DataFrame): True labels. Shape (N, 8).
            Order: [patient_overall, C1, C2, C3, C4, C5, C6, C7]
        y_pred (np.ndarray or pd.DataFrame): Predicted probabilities. Shape (N, 8).
            Order: [patient_overall, C1, C2, C3, C4, C5, C6, C7]
        eps (float): Epsilon for clipping predictions to avoid log(0).

    Returns:
        float: The weighted log loss.
    """
    # Convert to numpy arrays if DataFrames
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values

    # Ensure shapes match
    assert (
        y_true.shape == y_pred.shape
    ), f"Shape mismatch: {y_true.shape} vs {y_pred.shape}"

    # Clip predictions
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # Define weights
    # Column 0 is patient_overall, Columns 1-7 are C1-C7
    # Weights: Overall = 1.0, Vertebrae = 1/7
    weights = np.array([1.0] + [1.0 / 7.0] * 7)

    # Calculate binary cross entropy for each element
    # L = -w * [y * log(p) + (1-y) * log(1-p)]
    loss_matrix = -1.0 * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # Apply weights to columns
    weighted_loss_matrix = loss_matrix * weights

    # Average across all entries (as per competition metric description usually interpreted)
    # Note: The prompt says "loss is averaged across all rows".
    # In the submission format, each sub-type is a row.
    # So we sum all weighted losses and divide by the total number of predictions (rows in submission).
    # Since our matrix is (N_patients, 8), the total number of "rows" in submission is N_patients * 8.

    total_loss = np.sum(weighted_loss_matrix)
    num_predictions = y_true.size

    average_loss = total_loss / num_predictions

    return average_loss
