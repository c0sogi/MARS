import pandas as pd
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TARGET_COL,
)
from library.feature_engineering import generate_features


def load_dataset(dataset_type, load_cached_data=True, debug_n=None):
    """
    Loads the dataset for the specified type (train, val, test).

    This function delegates the heavy lifting of feature extraction and caching
    to the library.feature_engineering.generate_features function.
    It then formats the output into X (features) and y (target).

    Args:
        dataset_type (str): One of 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from the cache.
        debug_n (int, optional): Number of samples to process for debugging.

    Returns:
        X (pd.DataFrame): The feature matrix with segment_id as index.
        y (pd.Series or None): The target variable (time_to_eruption) or None for test set.
    """
    # 1. Determine the correct metadata path based on dataset type
    if dataset_type == "train":
        meta_path = TRAIN_METADATA_PATH
    elif dataset_type == "val":
        meta_path = VAL_METADATA_PATH
    elif dataset_type == "test":
        meta_path = TEST_METADATA_PATH
    else:
        raise ValueError(
            f"Invalid dataset_type: {dataset_type}. Must be 'train', 'val', or 'test'."
        )

    # 2. Generate features (Caching logic is handled inside generate_features)
    # This returns a DataFrame containing segment_id, features, and optionally the target.
    df = generate_features(
        metadata_path=meta_path,
        dataset_name=dataset_type,
        load_cached_data=load_cached_data,
        debug_n=debug_n,
    )

    # 3. Format the data for the model
    # Set segment_id as index so it's not used as a feature but is preserved
    if "segment_id" in df.columns:
        df = df.set_index("segment_id")

    # Separate Target and Features
    if TARGET_COL in df.columns:
        y = df[TARGET_COL]
        X = df.drop(columns=[TARGET_COL])
    else:
        # Test set case
        X = df
        y = None

    return X, y
