import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import matthews_corrcoef
from library.config import Config

# --------------------------------------------------------------------------
# Metric Evaluation
# --------------------------------------------------------------------------


def calculate_mcc(y_true, y_pred):
    """
    Calculates the Matthews Correlation Coefficient.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.

    Returns:
        float: The MCC score.
    """
    return matthews_corrcoef(y_true, y_pred)


# --------------------------------------------------------------------------
# I/O & Caching Utilities
# --------------------------------------------------------------------------


def save_parquet(df, path):
    """
    Saves a DataFrame to a parquet file, ensuring the directory exists.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path):
    """
    Loads a DataFrame from a parquet file.
    """
    return pd.read_parquet(path)


def save_npy(arr, path):
    """
    Saves a numpy array to a .npy file, ensuring the directory exists.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)


def load_npy(path):
    """
    Loads a numpy array from a .npy file.
    """
    return np.load(path)


def save_joblib(obj, path):
    """
    Saves a python object (e.g., model) using joblib, ensuring the directory exists.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path):
    """
    Loads a python object using joblib.
    """
    return joblib.load(path)


# --------------------------------------------------------------------------
# Memory Optimization
# --------------------------------------------------------------------------


def reduce_mem_usage(df):
    """
    Iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and str(col_type) != "category":
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
                    )  # float16 has low precision, safe bet is float32
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    # print(f'Memory usage reduced to {end_mem:.2f} MB ({(100 * (start_mem - end_mem) / start_mem):.1f}% reduction)')
    return df


# --------------------------------------------------------------------------
# Vectorized Physics Helpers
# --------------------------------------------------------------------------


def vectorized_distance(x1, y1, x2, y2):
    """
    Calculates Euclidean distance between two sets of coordinates.
    """
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def vectorized_speed(vx, vy):
    """
    Calculates speed magnitude from velocity components.
    """
    return np.sqrt(vx**2 + vy**2)


def vectorized_closing_speed(x1, y1, vx1, vy1, x2, y2, vx2, vy2):
    """
    Calculates the closing speed between two entities.
    Closing speed is the rate at which the distance between two points is decreasing.

    Formula: - (r_rel . v_rel) / |r_rel|
    Where r_rel = r2 - r1, v_rel = v2 - v1

    Positive closing speed means they are getting closer.
    Negative closing speed means they are moving apart.
    """
    # Relative position vector (from 1 to 2)
    rx = x2 - x1
    ry = y2 - y1

    # Relative velocity vector (from 1 to 2)
    rvx = vx2 - vx1
    rvy = vy2 - vy1

    # Distance squared
    dist_sq = rx**2 + ry**2

    # Dot product of relative position and relative velocity
    dot_prod = rx * rvx + ry * rvy

    # Rate of change of distance (time derivative of distance)
    # d(dist)/dt = (r . v) / |r|
    # We want closing speed, which is -d(dist)/dt

    # Handle division by zero (if distance is 0, closing speed is effectively 0 or undefined)
    # We use a small epsilon or mask
    dist = np.sqrt(dist_sq)

    # Initialize with zeros
    closing_speed = np.zeros_like(dist)

    # Only calculate where distance > 0
    mask = dist > 1e-6
    closing_speed[mask] = -(dot_prod[mask] / dist[mask])

    return closing_speed
