import os
import sys
import logging
import pandas as pd
import numpy as np
from library.config import Config


def setup_logger(name="pipeline", log_file=None, level=logging.INFO):
    """
    Configures and returns a logger instance that writes to stdout and optionally a file.

    Args:
        name (str): The name of the logger.
        log_file (str, optional): Path to a file where logs should be saved.
        level (int): Logging level (default: logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to prevent duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Stream Handler (Console)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # File Handler (Optional)
    if log_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def save_submission(
    ids,
    predictions,
    output_path=Config.SUBMISSION_PATH,
    id_col=Config.ID_COL,
    target_col=Config.TARGET_COL,
    mapping=None,
):
    """
    Formats predictions into a DataFrame and saves them to a CSV file.

    Args:
        ids (array-like): Sequence of sample IDs.
        predictions (array-like): Sequence of predicted class labels.
        output_path (str): File path to save the submission CSV.
        id_col (str): Name of the ID column.
        target_col (str): Name of the target column.
        mapping (dict, optional): Dictionary to map prediction values (e.g., inverse transform 0-indexed classes).
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert to numpy arrays for consistent handling
    ids_arr = np.array(ids)
    preds_arr = np.array(predictions)

    # Apply mapping if provided (e.g., converting 0-indexed model outputs back to original class labels)
    if mapping is not None:
        # Use pandas map for vectorization; fillna ensures unmapped values remain (though shouldn't happen)
        preds_arr = (
            pd.Series(preds_arr).map(mapping).fillna(pd.Series(preds_arr)).values
        )

    # Ensure integer type for targets if they are class labels
    if np.issubdtype(preds_arr.dtype, np.number):
        preds_arr = preds_arr.astype(int)

    # Create DataFrame
    df = pd.DataFrame({id_col: ids_arr, target_col: preds_arr})

    # Save to CSV
    df.to_csv(output_path, index=False)

    # Feedback
    print(f"Submission saved to: {output_path}")
    print(f"Submission shape: {df.shape}")
    print(f"First 5 rows:\n{df.head().to_string(index=False)}")
