import os
import random
import numpy as np
import pandas as pd
import joblib
import logging
import time
from sklearn.metrics import matthews_corrcoef
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and OS environments.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def setup_logger(log_file_path):
    """
    Configures a simple logger that writes to both a file and the console.
    """
    # Create directory if it doesn't exist
    log_dir = os.path.dirname(log_file_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],
    )
    return logging.getLogger()


def calc_mcc(y_true, y_pred):
    """
    Calculates the Matthews Correlation Coefficient.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        float: MCC score.
    """
    return matthews_corrcoef(y_true, y_pred)


def optimize_threshold(y_true, y_pred_probs, num_steps=100):
    """
    Finds the optimal probability threshold that maximizes the MCC score.

    Args:
        y_true: Ground truth binary labels.
        y_pred_probs: Predicted probabilities (0 to 1).
        num_steps: Number of threshold steps to evaluate.

    Returns:
        best_threshold (float): The threshold that maximizes MCC.
        best_score (float): The maximum MCC score achieved.
    """
    thresholds = np.linspace(0.01, 0.99, num_steps)
    best_threshold = 0.5
    best_score = -1.0

    # Pre-convert to numpy arrays for speed if they aren't already
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)

    for thresh in thresholds:
        y_pred_bin = (y_pred_probs >= thresh).astype(int)
        score = matthews_corrcoef(y_true, y_pred_bin)

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold, best_score


def save_model(model, path):
    """
    Saves a model object using joblib.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    """
    Loads a model object using joblib.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    return joblib.load(path)


def save_data(data, path):
    """
    Saves data to disk using Parquet (for DataFrames) or NPY (for NumPy arrays).
    Strictly avoids Pickle for data storage as per requirements.

    Args:
        data: pandas.DataFrame or numpy.ndarray
        path: Destination path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if isinstance(data, pd.DataFrame):
        if not path.endswith(".parquet"):
            path += ".parquet"
        data.to_parquet(path, index=False)
    elif isinstance(data, np.ndarray):
        if not path.endswith(".npy"):
            path += ".npy"
        np.save(path, data)
    else:
        # Fallback for other serializable objects (e.g. lists/dicts of metadata)
        # that are not heavy data arrays, using joblib is acceptable for metadata,
        # but the prompt prefers avoiding pickle for 'deterministic data processing'.
        # We assume this function is primarily for the heavy feature matrices.
        raise ValueError(
            "save_data only supports pandas DataFrame (parquet) or numpy array (npy)."
        )


def load_data(path):
    """
    Loads data from disk (Parquet or NPY).
    """
    if not os.path.exists(path):
        # Check if extensions were omitted
        if os.path.exists(path + ".parquet"):
            path += ".parquet"
        elif os.path.exists(path + ".npy"):
            path += ".npy"
        else:
            return None

    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.endswith(".npy"):
        return np.load(path)
    else:
        raise ValueError(f"Unsupported file extension for loading data: {path}")


class Timer:
    """
    Context manager to measure execution time.
    """

    def __init__(self, name="Process"):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        print(f"[{self.name}] Started...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        print(f"[{self.name}] Finished. Duration: {elapsed:.4f} seconds.")
