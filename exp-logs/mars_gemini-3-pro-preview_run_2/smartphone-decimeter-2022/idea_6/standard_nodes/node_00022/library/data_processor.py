import os
import numpy as np
import pandas as pd
from library.config import Config
from library.model import prepare_data, GNSSWindowDataset


def load_data(mode="train", load_cached_data=True, debug_size=None):
    """
    Loads and processes data for the specified mode (train, val, test).

    This function executes the data processing pipeline which includes:
    1. Reading GNSS and Ground Truth files based on the metadata CSVs.
    2. Aggregating raw GNSS measurements by epoch.
    3. Merging GNSS data with Ground Truth (or Test) timestamps.
    4. Feature Engineering: Converting ECEF coordinates to local ENU, calculating velocities.
    5. Windowing: Creating sliding windows of size Config.WINDOW_SIZE.
    6. Relative Centering: Transforming absolute coordinates in the window to be relative
       to the center frame, ensuring translation invariance.
    7. Scaling: Standardizing features (fit on train, transform on val/test).
    8. Caching: Saving processed data to disk to speed up future runs.

    Args:
        mode (str): One of 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load pre-processed data from cache.
                                 If False or cache is missing, re-processes from scratch.
        debug_size (int, optional): If provided, limits the dataset to this many samples
                                    for rapid debugging.

    Returns:
        tuple: (X, y, meta_df)
            X (np.ndarray): Input features tensor of shape (N, Window_Size, Channels).
            y (np.ndarray or None): Target residuals (Delta East, Delta North) of shape (N, 2).
                                    None if mode is 'test'.
            meta_df (pd.DataFrame): Metadata dataframe aligned with X, containing tripIds
                                    and timestamps.
    """
    # Select the appropriate metadata file based on the mode
    if mode == "train":
        metadata_path = Config.TRAIN_METADATA_PATH
    elif mode == "val":
        metadata_path = Config.VAL_METADATA_PATH
    elif mode == "test":
        metadata_path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'train', 'val', or 'test'.")

    # Delegate to the robust processing pipeline provided in library.model
    # This function handles the complex logic of windowing, relative centering, and caching.
    return prepare_data(
        metadata_path=metadata_path,
        mode=mode,
        load_cached_data=load_cached_data,
        debug_size=debug_size,
    )


def get_dataset(X, y=None):
    """
    Wraps the processed numpy arrays into a PyTorch Dataset.

    Args:
        X (np.ndarray): Input features.
        y (np.ndarray, optional): Target values.

    Returns:
        GNSSWindowDataset: A PyTorch Dataset instance compatible with DataLoader.
    """
    return GNSSWindowDataset(X, y)
