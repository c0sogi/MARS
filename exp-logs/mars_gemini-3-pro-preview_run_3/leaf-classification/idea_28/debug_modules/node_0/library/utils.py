import os
import random
import logging
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logger(name="leaf_identification"):
    """
    Configures and returns a logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        log_path = os.path.join(Config.WORKING_DIR, "execution.log")
        fh = logging.FileHandler(log_path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def load_data(split, load_cached_data=True, debug=Config.DEBUG):
    """
    Loads the dataset for a specific split (train, val, test).

    Implements a caching mechanism using Parquet files in Config.CACHE_DIR.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): If True, attempts to load from cache first.
        debug (bool): If True, limits the dataset size to Config.MAX_DEBUG_SAMPLES.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    # Define source paths based on split
    if split == "train":
        src_path = Config.TRAIN_METADATA
    elif split == "val":
        src_path = Config.VAL_METADATA
    elif split == "test":
        src_path = Config.TEST_METADATA
    else:
        raise ValueError(f"Unknown split: {split}")

    # Construct cache filename
    # Include debug flag in filename to separate full data from debug data
    debug_suffix = "_debug" if debug else ""
    cache_filename = f"data_{split}{debug_suffix}.parquet"
    cache_path = os.path.join(Config.CACHE_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # If loading fails, proceed to recompute/reload from source
            pass

    # 2. Load from source
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Metadata file not found at {src_path}")

    df = pd.read_csv(src_path)

    # Apply debug sampling
    if debug:
        if len(df) > Config.MAX_DEBUG_SAMPLES:
            df = df.sample(
                n=Config.MAX_DEBUG_SAMPLES, random_state=Config.SEED
            ).reset_index(drop=True)

    # 3. Save to cache
    # Ensure directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


def calculate_metric(y_true, y_pred, labels=None):
    """
    Calculates the Multi-class Log Loss metric.

    Follows the competition scoring logic:
    1. Rescales rows to sum to 1.
    2. Clips probabilities to [1e-15, 1-1e-15].
    3. Computes log loss.

    Args:
        y_true: Ground truth labels (array-like of shape (n_samples,)).
        y_pred: Predicted probabilities (array-like of shape (n_samples, n_classes)).
        labels: List of class labels to index the columns of y_pred.

    Returns:
        float: The log loss score.
    """
    y_pred = np.array(y_pred)

    # Rescale probabilities (each row divided by row sum)
    row_sums = y_pred.sum(axis=1)
    # Handle zero sums to avoid division by zero (though unlikely with valid model output)
    row_sums[row_sums == 0] = 1.0
    y_pred = y_pred / row_sums[:, np.newaxis]

    # Clip probabilities to avoid extremes of log function
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # Calculate log loss
    return log_loss(y_true, y_pred, labels=labels)


def save_submission(ids, probabilities, class_names, filename=Config.SUBMISSION_PATH):
    """
    Saves the predicted probabilities to a CSV file in the required format.

    Args:
        ids: List or array of image IDs.
        probabilities: Matrix of predicted probabilities.
        class_names: List of class names corresponding to probability columns.
        filename: Path to save the submission file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    df = pd.DataFrame(probabilities, columns=class_names)
    df.insert(0, "id", ids)

    df.to_csv(filename, index=False)
