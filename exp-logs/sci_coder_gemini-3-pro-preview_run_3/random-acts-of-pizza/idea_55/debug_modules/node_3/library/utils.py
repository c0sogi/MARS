import os
import random
import logging
import numpy as np
import pandas as pd
import joblib
import torch
from library.config import Config


def set_seed(seed=Config.RANDOM_STATE):
    """
    Sets the random seed for reproducibility across Python, NumPy, and Torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def setup_logging(log_name="pipeline.log", level=logging.INFO):
    """
    Configures the root logger to output to console and file.
    """
    log_path = os.path.join(Config.WORKING_DIR, log_name)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()],
    )
    logging.info(f"Logging initialized. Log file: {log_path}")


def get_model_path(name, is_volatile=False, fold=None):
    """
    Constructs the file path for a model artifact.
    """
    model_dir = os.path.join(Config.WORKING_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)

    if is_volatile:
        if fold is None:
            raise ValueError("Fold must be specified for volatile models.")
        filename = f"{name}_fold_{fold}.joblib"
    else:
        filename = f"{name}.joblib"

    return os.path.join(model_dir, filename)


def save_model(model, name, is_volatile=False, fold=None):
    """
    Saves a model artifact using joblib.
    """
    path = get_model_path(name, is_volatile, fold)
    joblib.dump(model, path)
    logging.info(f"Model saved to {path}")


def load_model(name, is_volatile=False, fold=None):
    """
    Loads a model artifact using joblib.
    """
    path = get_model_path(name, is_volatile, fold)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")

    logging.info(f"Loading model from {path}")
    return joblib.load(path)


def save_submission(df):
    """
    Saves the submission DataFrame to the configured path.
    """
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    df.to_csv(Config.SUBMISSION_PATH, index=False)
    logging.info(f"Submission saved to {Config.SUBMISSION_PATH}")


def load_or_process_data(
    file_name, process_fn, load_cache=True, file_type="parquet", **kwargs
):
    """
    Generic caching mechanism for data processing.

    Args:
        file_name (str): Name of the file (relative to WORKING_DIR).
        process_fn (callable): Function to compute data if cache is missed.
        load_cache (bool): Whether to attempt loading from cache.
        file_type (str): 'parquet' for DataFrame, 'npy' or 'npz' for NumPy.
        **kwargs: Arguments passed to process_fn.

    Returns:
        The loaded or computed data.
    """
    file_path = os.path.join(Config.WORKING_DIR, file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 1. Attempt Load
    if load_cache and os.path.exists(file_path):
        logging.info(f"Loading cached data from {file_path}")
        try:
            if file_type == "parquet":
                return pd.read_parquet(file_path)
            elif file_type == "npy":
                return np.load(file_path)
            elif file_type == "npz":
                return np.load(file_path)  # Returns NpzFile object
            else:
                raise ValueError(f"Unsupported file_type: {file_type}")
        except Exception as e:
            logging.warning(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute
    logging.info(f"Computing data for {file_name}...")
    data = process_fn(**kwargs)

    # 3. Save
    logging.info(f"Saving data to {file_path}")
    if file_type == "parquet":
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Expected pandas DataFrame for parquet format.")
        data.to_parquet(file_path, index=False)
    elif file_type == "npy":
        np.save(file_path, data)
    elif file_type == "npz":
        # Assumes data is a dict or kwargs for savez
        if isinstance(data, dict):
            np.savez(file_path, **data)
        else:
            # If it's a tuple/list, we can't easily name them without more info,
            # so we assume the user returns a dict for npz or handles saving inside process_fn
            # (but this function is designed to handle saving).
            # Fallback for simple array lists
            np.savez(file_path, *data)
    else:
        raise ValueError(f"Unsupported file_type: {file_type}")

    return data
