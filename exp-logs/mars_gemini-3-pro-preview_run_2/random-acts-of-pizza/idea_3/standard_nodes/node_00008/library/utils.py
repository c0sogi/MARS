import sys
import os
import time
import logging
import pandas as pd
from contextlib import contextmanager
from library.config import set_seed as _set_seed


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility.
    Wraps the implementation from library.config to ensure consistency.
    """
    _set_seed(seed)


def setup_logger(name="model_logger", level=logging.INFO):
    """
    Configures and returns a logger instance that outputs to stdout.
    Ensures handlers are not duplicated if the logger is retrieved multiple times.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times if logger is reused
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


@contextmanager
def timer(name, logger=None):
    """
    Context manager to measure and log the execution time of a code block.

    Args:
        name (str): The name of the operation being timed.
        logger (logging.Logger, optional): Logger instance to use. If None, prints to stdout.
    """
    t0 = time.time()
    start_msg = f"[{name}] Start"
    if logger:
        logger.info(start_msg)
    else:
        print(start_msg)

    yield

    elapsed = time.time() - t0
    end_msg = f"[{name}] Done in {elapsed:.2f} s"
    if logger:
        logger.info(end_msg)
    else:
        print(end_msg)


def save_submission(request_ids, predictions, output_path):
    """
    Saves the predictions to a CSV file in the competition's required format.

    Args:
        request_ids (list or array): List of request IDs corresponding to the test set.
        predictions (list or array): List of predicted probabilities (real-valued).
        output_path (str): Full path where the submission CSV should be saved.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame matching the submission format
    submission_df = pd.DataFrame(
        {"request_id": request_ids, "requester_received_pizza": predictions}
    )

    # Save to CSV without the index
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
