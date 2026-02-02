import pandas as pd
import numpy as np
import os
from library.config import Config


def load_metadata(split="train"):
    """
    Loads the metadata CSV for the specified split.

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: Metadata dataframe.
    """
    if split == "train":
        path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        path = Config.VAL_METADATA_PATH
    elif split == "test":
        path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid split: {split}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    return pd.read_csv(path)


def load_gnss(path):
    """
    Loads GNSS data from the specified path.
    Selects columns defined in Config.GNSS_COLS.
    Performs type casting and basic cleaning.

    Args:
        path (str): Relative path to the device_gnss.csv file.

    Returns:
        pd.DataFrame: Processed GNSS dataframe.
    """
    full_path = os.path.join(Config.INPUT_DIR, path)
    if not os.path.exists(full_path):
        # Return empty DataFrame with expected columns if file missing
        return pd.DataFrame(columns=Config.GNSS_COLS)

    # Read only necessary columns to save memory
    try:
        df = pd.read_csv(
            full_path,
            usecols=Config.GNSS_COLS,
            dtype={
                "Cn0DbHz": "float32",
                "SvElevationDegrees": "float32",
                "Svid": "int32",
                "utcTimeMillis": "int64",
            },
        )
    except ValueError:
        # Fallback if usecols fails (e.g. column missing)
        df = pd.read_csv(full_path)
        # Ensure all config columns exist
        for col in Config.GNSS_COLS:
            if col not in df.columns:
                df[col] = np.nan
        df = df[Config.GNSS_COLS]

    # Basic Cleaning
    # Fill NaNs in signal strength with 0 (weakest)
    if "Cn0DbHz" in df.columns:
        df["Cn0DbHz"] = df["Cn0DbHz"].fillna(0).astype("float32")

    # Fill NaNs in Elevation with 0 (horizon)
    if "SvElevationDegrees" in df.columns:
        df["SvElevationDegrees"] = df["SvElevationDegrees"].fillna(0).astype("float32")

    return df


def load_imu(path):
    """
    Loads IMU data from the specified path.
    Selects columns defined in Config.IMU_COLS.
    Performs type casting.

    Args:
        path (str): Relative path to the device_imu.csv file.

    Returns:
        pd.DataFrame: Processed IMU dataframe.
    """
    full_path = os.path.join(Config.INPUT_DIR, path)
    if not os.path.exists(full_path):
        return pd.DataFrame(columns=Config.IMU_COLS)

    try:
        df = pd.read_csv(
            full_path,
            usecols=Config.IMU_COLS,
            dtype={
                "MeasurementX": "float32",
                "MeasurementY": "float32",
                "MeasurementZ": "float32",
                "utcTimeMillis": "int64",
                "MessageType": "object",
            },
        )
    except ValueError:
        df = pd.read_csv(full_path)
        for col in Config.IMU_COLS:
            if col not in df.columns:
                df[col] = np.nan
        df = df[Config.IMU_COLS]

    return df


def load_ground_truth(path):
    """
    Loads Ground Truth data from the specified path.

    Args:
        path (str): Relative path to the ground_truth.csv file.

    Returns:
        pd.DataFrame: Ground Truth dataframe.
    """
    full_path = os.path.join(Config.INPUT_DIR, path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Ground truth file not found: {full_path}")

    df = pd.read_csv(full_path)

    # Ensure timestamp type matches GNSS/IMU for merging
    if "UnixTimeMillis" in df.columns:
        df["UnixTimeMillis"] = df["UnixTimeMillis"].astype("int64")

    return df
