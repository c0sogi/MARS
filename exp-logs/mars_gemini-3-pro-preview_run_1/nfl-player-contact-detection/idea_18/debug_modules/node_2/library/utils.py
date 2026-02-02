import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from library.config import Config


def setup_logger(name="idea_18", log_file=None):
    """
    Configures and returns a logger with console and optional file output.
    Ensures handlers are not duplicated if the logger is retrieved multiple times.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if function is called repeatedly
    if not logger.handlers:
        # Console Handler
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setLevel(logging.INFO)
        c_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        c_handler.setFormatter(c_format)
        logger.addHandler(c_handler)

        # File Handler (Optional)
        if log_file:
            # Ensure directory for log file exists
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            f_handler = logging.FileHandler(log_file)
            f_handler.setLevel(logging.INFO)
            f_format = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            f_handler.setFormatter(f_format)
            logger.addHandler(f_handler)

    return logger


def compute_mcc(y_true, y_pred):
    """
    Computes the Matthews Correlation Coefficient between ground truth and predictions.
    Expects binary labels (0/1) for both inputs.
    """
    return matthews_corrcoef(y_true, y_pred)


def save_model(model, filename):
    """
    Saves a trained model to the configured model directory using joblib.
    """
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    file_path = os.path.join(Config.MODEL_DIR, filename)
    joblib.dump(model, file_path)


def load_model(filename):
    """
    Loads a trained model from the configured model directory using joblib.
    """
    file_path = os.path.join(Config.MODEL_DIR, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found at: {file_path}")
    return joblib.load(file_path)


def save_cache_parquet(df, filename):
    """
    Saves a pandas DataFrame to a Parquet file in the configured cache directory.
    Used for caching intermediate dataframes deterministically without pickle.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    file_path = os.path.join(Config.CACHE_DIR, filename)
    df.to_parquet(file_path, index=False)


def load_cache_parquet(filename):
    """
    Loads a pandas DataFrame from a Parquet file in the configured cache directory.
    Returns None if the file does not exist.
    """
    file_path = os.path.join(Config.CACHE_DIR, filename)
    if os.path.exists(file_path):
        return pd.read_parquet(file_path)
    return None


def save_cache_npy(arr, filename):
    """
    Saves a numpy array to a .npy file in the configured cache directory.
    Used for caching intermediate arrays deterministically.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    file_path = os.path.join(Config.CACHE_DIR, filename)
    np.save(file_path, arr)


def load_cache_npy(filename):
    """
    Loads a numpy array from a .npy file in the configured cache directory.
    Returns None if the file does not exist.
    """
    file_path = os.path.join(Config.CACHE_DIR, filename)
    if os.path.exists(file_path):
        return np.load(file_path)
    return None
