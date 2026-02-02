import os
import pandas as pd
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    SAMPLE_SUBMISSION_PATH,
)
from library.features import generate_dataset, process_drive


def load_drive_data(
    drive_id, phone_name, gnss_rel, imu_rel, gt_rel=None, is_train=True
):
    """
    Load and process data for a specific drive.
    Wrapper around library.features.process_drive.

    Args:
        drive_id (str): The drive identifier.
        phone_name (str): The phone model name.
        gnss_rel (str): Relative path to GNSS file.
        imu_rel (str): Relative path to IMU file.
        gt_rel (str, optional): Relative path to Ground Truth file.
        is_train (bool): Whether this is for training (calculates targets).

    Returns:
        pd.DataFrame: Processed dataframe for the drive.
    """
    return process_drive(drive_id, phone_name, gnss_rel, imu_rel, gt_rel, is_train)


def load_train_data(load_cached_data=True, limit=None):
    """
    Load the training dataset with features and targets.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache.
                                 If False or cache missing, regenerates from source.
        limit (int, optional): If provided, returns only the first N rows (for debugging).

    Returns:
        pd.DataFrame: Training data containing features and ENU residual targets.
    """
    df = generate_dataset(
        TRAIN_METADATA_PATH, "train", load_cached_data=load_cached_data
    )
    if limit is not None:
        return df.head(limit)
    return df


def load_val_data(load_cached_data=True, limit=None):
    """
    Load the validation dataset with features and targets.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache.
        limit (int, optional): If provided, returns only the first N rows.

    Returns:
        pd.DataFrame: Validation data containing features and ENU residual targets.
    """
    df = generate_dataset(VAL_METADATA_PATH, "val", load_cached_data=load_cached_data)
    if limit is not None:
        return df.head(limit)
    return df


def load_test_data(load_cached_data=True, limit=None):
    """
    Load the test dataset with features.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache.
        limit (int, optional): If provided, returns only the first N rows.

    Returns:
        pd.DataFrame: Test data containing features and WLS baseline positions.
    """
    df = generate_dataset(TEST_METADATA_PATH, "test", load_cached_data=load_cached_data)
    if limit is not None:
        return df.head(limit)
    return df


def load_sample_submission():
    """
    Load the sample submission file to get the required output format.

    Returns:
        pd.DataFrame: Sample submission dataframe.
    """
    if not os.path.exists(SAMPLE_SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Sample submission file not found at {SAMPLE_SUBMISSION_PATH}"
        )
    return pd.read_csv(SAMPLE_SUBMISSION_PATH)
