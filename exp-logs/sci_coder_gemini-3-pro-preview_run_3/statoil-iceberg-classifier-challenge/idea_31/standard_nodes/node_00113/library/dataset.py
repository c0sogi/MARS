import os
import numpy as np
import pandas as pd
from library.utils import process_data, IcebergDataset, set_seed


def load_data(base_dir="./working/idea_31", load_cached_data=True, dataset_size=None):
    """
    Loads and processes the dataset for the Iceberg vs Ship classification task.

    This function leverages the pre-implemented process_data utility to:
    1. Load raw JSON data (train.json, test.json).
    2. Construct 3-channel images (Band 1, Band 2, Average).
    3. Impute missing incidence angles using the training set median.
    4. Cache the processed arrays to disk for faster subsequent access.

    Args:
        base_dir (str): Directory where cached .npy files are stored.
        load_cached_data (bool): If True, attempts to load from cache first.
        dataset_size (int, optional): If provided, truncates the data to this size
                                      for debugging purposes.

    Returns:
        tuple: (X_train, y_train, angle_train, X_test, ids_test, angle_test)
    """
    # Ensure reproducibility
    set_seed(42)

    # Ensure the working directory exists
    os.makedirs(base_dir, exist_ok=True)

    # Call the library function to handle data processing and caching
    # process_data handles the logic of checking cache, loading JSONs,
    # processing images/angles, and saving cache.
    X_train, y_train, angle_train, X_test, ids_test, angle_test = process_data(
        load_cached_data=load_cached_data, base_dir=base_dir
    )

    # Handle dataset_size for debugging/testing
    if dataset_size is not None:
        if len(X_train) > dataset_size:
            X_train = X_train[:dataset_size]
            y_train = y_train[:dataset_size]
            angle_train = angle_train[:dataset_size]

        if len(X_test) > dataset_size:
            X_test = X_test[:dataset_size]
            ids_test = ids_test[:dataset_size]
            angle_test = angle_test[:dataset_size]

    return X_train, y_train, angle_train, X_test, ids_test, angle_test
