import os
import sys
import logging
import random
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import matthews_corrcoef
from library import config


def setup_logging(log_path=None, level=logging.INFO):
    """
    Configures the logging configuration for the application.

    Args:
        log_path (str, optional): Path to the log file. If None, logs only to stdout.
        level (int): Logging level (default: logging.INFO).
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


def seed_everything(seed=config.SEED):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    # Note: Torch seeding would go here if torch was used,
    # but the prompt focuses on sklearn/gbdt models.


def reduce_mem_usage(df, verbose=True):
    """
    Iterates through all the columns of a dataframe and modifies the data type
    to reduce memory usage.

    Args:
        df (pd.DataFrame): The dataframe to optimize.
        verbose (bool): Whether to print the memory reduction statistics.

    Returns:
        pd.DataFrame: The optimized dataframe.
    """
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and col_type.name != "category":
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        logging.info(f"Memory usage of dataframe is {start_mem:.2f} MB")
        logging.info(f"Memory usage after optimization is: {end_mem:.2f} MB")
        logging.info(f"Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%")

    return df


def calc_mcc(y_true, y_pred):
    """
    Calculates the Matthews Correlation Coefficient.

    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets as returned by a classifier.

    Returns:
        float: The MCC score.
    """
    return matthews_corrcoef(y_true, y_pred)


def save_model(model, path):
    """
    Saves a model to disk using joblib.

    Args:
        model: The model object to save.
        path (str): The destination path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logging.info(f"Model saved to {path}")


def load_model(path):
    """
    Loads a model from disk using joblib.

    Args:
        path (str): The path to the model file.

    Returns:
        The loaded model object.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    model = joblib.load(path)
    logging.info(f"Model loaded from {path}")
    return model
