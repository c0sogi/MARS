import os
import pandas as pd
import numpy as np
from library.config import INPUT_DIR, METADATA_DIR


def load_metadata(
    split_name: str, sample_size: int = None, random_state: int = 42
) -> pd.DataFrame:
    """
    Loads the metadata CSV for a specific split (train, val, or test).

    Args:
        split_name (str): The name of the split ('train', 'val', or 'test').
        sample_size (int, optional): If provided, returns a random sample of this size.
                                     Useful for debugging or quick pipeline checks.
        random_state (int): Seed for reproducibility when sampling.

    Returns:
        pd.DataFrame: DataFrame containing segment_ids, file_paths, and targets (if available).
    """
    valid_splits = ["train", "val", "test"]
    if split_name not in valid_splits:
        raise ValueError(
            f"Invalid split_name: {split_name}. Must be one of {valid_splits}"
        )

    file_path = os.path.join(METADATA_DIR, f"{split_name}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Metadata file not found at {file_path}")

    df = pd.read_csv(file_path)

    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    return df


def load_sensor_segment(file_path: str, fill_na: bool = True) -> pd.DataFrame:
    """
    Loads a single sensor segment CSV file from the input directory.

    Args:
        file_path (str): Relative path to the CSV file (e.g., 'train/1000015382.csv').
        fill_na (bool): If True, imputes missing values with the column-wise mean
                        to preserve DC offsets.

    Returns:
        pd.DataFrame: DataFrame containing the sensor readings (float32).
    """
    full_path = os.path.join(INPUT_DIR, file_path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Sensor file not found at {full_path}")

    # Load as float32 to handle NaNs and save memory
    df = pd.read_csv(full_path, dtype="float32")

    if fill_na:
        # Calculate mean per column (ignoring NaNs) to preserve DC offset
        column_means = df.mean()

        # Fill NaNs with the calculated means.
        # If a column is entirely NaN, mean() is NaN. We fill those residuals with 0.
        df = df.fillna(column_means).fillna(0.0)

    return df
