import os
import sys
import random
import logging
import numpy as np
import pandas as pd
import torch
import joblib
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and Torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(name="logger", log_file=None, level=logging.INFO):
    """
    Configures a logger to output to both the console and a file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to prevent duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console Handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def save_model(model, filename):
    """
    Saves a model object (e.g., sklearn pipeline) using joblib to the models cache directory.
    """
    filename = os.path.basename(filename)
    path = os.path.join(Config.WORKING_DIR, "models", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(filename):
    """
    Loads a model object from the models cache directory.
    """
    filename = os.path.basename(filename)
    path = os.path.join(Config.WORKING_DIR, "models", filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    return joblib.load(path)


def save_numpy(array, filename):
    """
    Saves a numpy array to the features cache directory as a .npy file.
    """
    filename = os.path.basename(filename)
    if not filename.endswith(".npy"):
        filename += ".npy"
    path = os.path.join(Config.WORKING_DIR, "features", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, array)


def load_numpy(filename):
    """
    Loads a numpy array from the features cache directory.
    Returns None if the file does not exist.
    """
    filename = os.path.basename(filename)
    if not filename.endswith(".npy"):
        filename += ".npy"
    path = os.path.join(Config.WORKING_DIR, "features", filename)
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)
    return None


def save_parquet(df, filename):
    """
    Saves a pandas DataFrame to the features cache directory as a .parquet file.
    """
    filename = os.path.basename(filename)
    if not filename.endswith(".parquet"):
        filename += ".parquet"
    path = os.path.join(Config.WORKING_DIR, "features", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(filename):
    """
    Loads a pandas DataFrame from the features cache directory.
    Returns None if the file does not exist.
    """
    filename = os.path.basename(filename)
    if not filename.endswith(".parquet"):
        filename += ".parquet"
    path = os.path.join(Config.WORKING_DIR, "features", filename)
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None
