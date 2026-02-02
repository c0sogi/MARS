import logging
import sys
import os
import numpy as np
import pandas as pd
from library.config import SUBMISSION_PATH


def setup_logger(
    name: str = "model_logger", log_file: str = None, level: int = logging.INFO
) -> logging.Logger:
    """
    Configures and returns a logger instance that writes to stdout and an optional file.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file.
        level (int): Logging level (default: logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to prevent duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def calculate_rmsle(y_true, y_pred):
    """
    Calculates the Column-wise Root Mean Squared Logarithmic Error (RMSLE).

    The metric is defined as the root mean squared error of the log-transformed
    predicted and true values. Since there are multiple targets, the final score
    is the average of the RMSLE for each target column.

    Args:
        y_true (array-like): Ground truth values.
        y_pred (array-like): Predicted values.

    Returns:
        float: The mean RMSLE across all target columns.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Clip predictions to be non-negative as log is undefined for negative numbers.
    # Formation energy and bandgap energy should physically be non-negative or close to it.
    y_true_clipped = np.maximum(0, y_true)
    y_pred_clipped = np.maximum(0, y_pred)

    # Calculate squared logarithmic error: (log(1+y) - log(1+y_pred))^2
    squared_log_error = (np.log1p(y_true_clipped) - np.log1p(y_pred_clipped)) ** 2

    # Calculate Mean Squared Logarithmic Error for each column
    if squared_log_error.ndim > 1:
        msle = np.mean(squared_log_error, axis=0)
    else:
        msle = np.mean(squared_log_error)

    # Calculate Root Mean Squared Logarithmic Error
    rmsle = np.sqrt(msle)

    # Return the average RMSLE across columns
    if np.ndim(rmsle) > 0:
        return np.mean(rmsle)
    else:
        return rmsle


def save_submission(ids, formation_energy, bandgap_energy, filename=SUBMISSION_PATH):
    """
    Saves the predictions to a CSV file in the format required for submission.

    Args:
        ids (array-like): List of sample IDs.
        formation_energy (array-like): Predicted formation energies.
        bandgap_energy (array-like): Predicted bandgap energies.
        filename (str): Output file path. Defaults to SUBMISSION_PATH from config.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Create DataFrame
    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": formation_energy,
            "bandgap_energy_ev": bandgap_energy,
        }
    )

    # Ensure correct column order
    cols = ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
    submission_df = submission_df[cols]

    # Save to CSV
    submission_df.to_csv(filename, index=False)
    print(f"Submission saved to {filename}")
