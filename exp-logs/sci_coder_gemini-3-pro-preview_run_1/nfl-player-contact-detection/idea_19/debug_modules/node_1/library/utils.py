import os
import sys
import logging
import random
import hashlib
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.metrics import matthews_corrcoef


def setup_logging(log_path=None, level=logging.INFO):
    """
    Configures the logging module with stdout and optional file handlers.
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_hash(obj):
    """
    Generates a unique MD5 hash for a given object (e.g., dictionary of parameters).
    """
    return hashlib.md5(str(obj).encode("utf-8")).hexdigest()


def calc_mcc(y_true, y_pred):
    """
    Calculates the Matthews Correlation Coefficient between truth and predictions.
    """
    return matthews_corrcoef(y_true, y_pred)


def save_parquet(df, path):
    """
    Saves a pandas DataFrame to a parquet file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path):
    """
    Loads a pandas DataFrame from a parquet file.
    """
    return pd.read_parquet(path)


def save_numpy(arr, path):
    """
    Saves a numpy array to a .npy file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)


def load_numpy(path):
    """
    Loads a numpy array from a .npy file.
    """
    return np.load(path)


def save_joblib(obj, path):
    """
    Saves an object using joblib (preferred for models).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path):
    """
    Loads an object using joblib.
    """
    return joblib.load(path)


# Aliases for compatibility with expected interface, using joblib backend
save_pickle = save_joblib
load_pickle = load_joblib
