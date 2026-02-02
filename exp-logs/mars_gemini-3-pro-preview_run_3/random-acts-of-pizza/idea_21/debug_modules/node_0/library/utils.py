import os
import sys
import random
import time
import logging
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from library import config


def set_seed(seed=config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logger(name="pipeline"):
    """
    Creates and configures a logger that outputs to stdout.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    # Prevent propagation to avoid double logging if root logger is configured
    logger.propagate = False
    return logger


class Timer:
    """
    Context manager to measure and log the execution time of a code block.
    """

    def __init__(self, description="Process"):
        self.description = description
        self.logger = get_logger("Timer")
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting: {self.description}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            self.logger.info(f"Finished: {self.description} | Time: {elapsed:.4f}s")


def load_dataset(split, sample_size=None):
    """
    Loads the dataset for the specified split ('train', 'val', 'test') from the metadata directory.

    Args:
        split (str): One of 'train', 'val', or 'test'.
        sample_size (int, optional): Number of samples to load for debugging/testing.
                                     If None or <= 0, loads the full dataset.

    Returns:
        pd.DataFrame: The loaded pandas DataFrame.
    """
    logger = get_logger("DataLoader")

    if split == "train":
        path = config.TRAIN_PATH
    elif split == "val":
        path = config.VAL_PATH
    elif split == "test":
        path = config.TEST_PATH
    else:
        raise ValueError(
            f"Invalid split '{split}'. Expected 'train', 'val', or 'test'."
        )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}")

    logger.info(f"Loading {split} data from {path}...")
    df = pd.read_parquet(path)

    if sample_size is not None and sample_size > 0:
        if sample_size < len(df):
            logger.info(f"Subsampling {sample_size} rows from {len(df)} total rows.")
            df = df.sample(n=sample_size, random_state=config.SEED).reset_index(
                drop=True
            )
        else:
            logger.info(
                f"Requested sample size {sample_size} >= total rows {len(df)}. Returning full dataset."
            )

    logger.info(f"Successfully loaded {len(df)} rows for {split} split.")
    return df


def print_metrics(y_true, y_pred_proba, split_name="Validation"):
    """
    Calculates and prints the ROC AUC metric with full precision.

    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred_proba (array-like): Predicted probabilities for the positive class.
        split_name (str): Name of the data split (e.g., 'Validation').

    Returns:
        float: The calculated ROC AUC score.
    """
    # Check if we have at least two classes to calculate AUC
    if len(np.unique(y_true)) < 2:
        print(f"[{split_name}] ROC AUC: Undefined (Only one class present in targets)")
        return 0.0

    auc_score = roc_auc_score(y_true, y_pred_proba)
    # Printing full precision as requested
    print(f"[{split_name}] ROC AUC: {auc_score}")
    return auc_score


def save_submission(request_ids, predictions, output_path=config.SUBMISSION_PATH):
    """
    Saves the final predictions to a CSV file in the required format.

    Args:
        request_ids (array-like): Sequence of request IDs.
        predictions (array-like): Sequence of predicted probabilities.
        output_path (str): File path to save the submission CSV.
    """
    logger = get_logger("Submission")

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    submission_df = pd.DataFrame(
        {config.ID_COL: request_ids, config.TARGET_COL: predictions}
    )

    submission_df.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}. Shape: {submission_df.shape}")
