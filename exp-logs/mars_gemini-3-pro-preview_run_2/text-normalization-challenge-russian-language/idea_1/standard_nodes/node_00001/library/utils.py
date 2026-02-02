import os
import sys
import logging
import pandas as pd
from library.config import SUBMISSION_FILE_PATH, COL_ID, COL_AFTER


def setup_logger(name="text_normalization", level=logging.INFO):
    """
    Configures and returns a logger instance for tracking progress.

    Args:
        name (str): The name of the logger.
        level (int): The logging level (e.g., logging.INFO).

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if handlers exist to avoid duplicate logs on re-execution
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def save_submission(predictions, filepath=SUBMISSION_FILE_PATH):
    """
    Saves the predictions to a CSV file in the format required for submission.

    Args:
        predictions (dict or pd.DataFrame):
            - If dict: Keys should be 'id' strings and values 'after' strings.
            - If pd.DataFrame: Must contain columns 'id' and 'after'.
        filepath (str): The path where the submission CSV will be saved.
                        Defaults to the path defined in config.
    """
    logger = setup_logger()

    # Normalize input to DataFrame
    if isinstance(predictions, dict):
        df = pd.DataFrame(list(predictions.items()), columns=[COL_ID, COL_AFTER])
    elif isinstance(predictions, pd.DataFrame):
        df = predictions.copy()
        # Validate or fix columns
        if COL_ID not in df.columns or COL_AFTER not in df.columns:
            # If strictly 2 columns, assume order is [id, after]
            if len(df.columns) == 2:
                df.columns = [COL_ID, COL_AFTER]
            else:
                raise ValueError(
                    f"DataFrame must contain '{COL_ID}' and '{COL_AFTER}' columns."
                )
    else:
        raise TypeError(
            "Predictions must be provided as a dictionary or a pandas DataFrame."
        )

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save to CSV
    # index=False removes the pandas row index
    # Standard to_csv handles quoting of special characters (like commas in text) automatically
    df.to_csv(filepath, index=False)

    logger.info(f"Submission saved successfully to {filepath}")
