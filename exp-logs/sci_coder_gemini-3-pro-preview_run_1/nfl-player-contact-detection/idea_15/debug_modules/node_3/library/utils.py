import os
import sys
import random
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import matthews_corrcoef
import torch

# Import configuration to access paths and constants
from library import config


def seed_everything(seed=config.RANDOM_STATE):
    """
    Sets the random seed for various libraries to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # Torch seeding if available
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def setup_logging(log_filename="execution.log"):
    """
    Configures the logging to write to console and a file in the working directory.
    """
    log_path = os.path.join(config.WORKING_DIR, log_filename)

    # Reset any existing handlers
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    logging.info(f"Logging initialized. Output writing to {log_path}")


def calc_mcc(y_true, y_pred, threshold=0.5):
    """
    Calculates the Matthews Correlation Coefficient.

    Args:
        y_true: Array-like of ground truth labels (binary).
        y_pred: Array-like of predictions. Can be probabilities or binary.
        threshold: Float threshold to convert probabilities to binary if needed.

    Returns:
        float: The MCC score.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # If predictions are probabilities (floats), apply threshold
    if np.issubdtype(y_pred.dtype, np.floating):
        y_pred_binary = (y_pred >= threshold).astype(int)
    else:
        y_pred_binary = y_pred.astype(int)

    return matthews_corrcoef(y_true, y_pred_binary)


def reduce_mem_usage(df):
    """
    Iterates through all the columns of a dataframe and modifies the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    logging.info(f"Memory usage of dataframe is {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype

        if (
            col_type != object
            and col_type.name != "category"
            and "datetime" not in col_type.name
        ):
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    # float16 has lower precision, using float32 is safer for ML
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage().sum() / 1024**2
    logging.info(f"Memory usage after optimization is {end_mem:.2f} MB")
    logging.info(f"Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%")

    return df


def save_model(model, filename):
    """
    Saves a model to the working directory using joblib.
    """
    models_dir = os.path.join(config.WORKING_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    file_path = os.path.join(models_dir, filename)
    joblib.dump(model, file_path)
    logging.info(f"Model saved to {file_path}")


def load_model(filename):
    """
    Loads a model from the working directory using joblib.
    """
    file_path = os.path.join(config.WORKING_DIR, "models", filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found at {file_path}")

    model = joblib.load(file_path)
    logging.info(f"Model loaded from {file_path}")
    return model
