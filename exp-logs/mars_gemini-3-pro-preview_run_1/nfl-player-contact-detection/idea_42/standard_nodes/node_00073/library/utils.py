import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef
from library.config import WORKING_DIR, SEED


def seed_everything(seed=SEED):
    """
    Sets the random seed for reproducibility across various libraries.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reduce_mem_usage(df):
    """
    Iterates through all the columns of a dataframe and modifies the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
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
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(
                        np.float32
                    )  # float16 has low precision, safe to use float32
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print(
        f"Memory usage reduced to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)"
    )
    return df


def calc_mcc(y_true, y_pred):
    """
    Calculates the Matthews Correlation Coefficient.
    """
    return matthews_corrcoef(y_true, y_pred)


def optimize_threshold(y_true, y_pred_proba, steps=100):
    """
    Finds the optimal probability threshold that maximizes the MCC score.

    Args:
        y_true: Ground truth binary labels.
        y_pred_proba: Predicted probabilities.
        steps: Number of threshold steps to scan.

    Returns:
        best_threshold: The threshold value that yielded the highest MCC.
        best_score: The highest MCC score achieved.
    """
    best_score = -1.0
    best_threshold = 0.5

    # Search range from 0.01 to 0.99
    thresholds = np.linspace(0.01, 0.99, steps)

    for thresh in thresholds:
        y_pred_binary = (y_pred_proba >= thresh).astype(int)
        score = matthews_corrcoef(y_true, y_pred_binary)

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold, best_score


def cache_result(filename, file_format="parquet"):
    """
    Decorator to cache function results to disk.

    Args:
        filename: Name of the file to save/load.
        file_format: 'parquet' for pandas DataFrames or 'npy' for numpy arrays.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            load_cached_data = kwargs.get("load_cached_data", False)
            filepath = os.path.join(WORKING_DIR, filename)

            # Ensure directory exists
            os.makedirs(WORKING_DIR, exist_ok=True)

            # 1. Try to load
            if load_cached_data:
                if os.path.exists(filepath):
                    print(f"[Cache] Loading {filename} from {filepath}...")
                    if file_format == "parquet":
                        return pd.read_parquet(filepath)
                    elif file_format == "npy":
                        return np.load(filepath, allow_pickle=True)
                    else:
                        raise ValueError(f"Unsupported format: {file_format}")
                else:
                    print(f"[Cache] File {filename} not found. Computing...")

            # 2. Compute
            result = func(*args, **kwargs)

            # 3. Save
            print(f"[Cache] Saving {filename} to {filepath}...")
            if file_format == "parquet":
                if isinstance(result, pd.DataFrame):
                    result.to_parquet(filepath, index=False)
                else:
                    raise TypeError("Result must be a DataFrame for parquet format.")
            elif file_format == "npy":
                np.save(filepath, result)
            else:
                raise ValueError(f"Unsupported format: {file_format}")

            return result

        return wrapper

    return decorator
