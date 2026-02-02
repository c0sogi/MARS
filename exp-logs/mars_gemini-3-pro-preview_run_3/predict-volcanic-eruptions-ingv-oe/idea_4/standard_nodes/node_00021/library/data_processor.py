import pandas as pd
import os
from library.feature_engineering import get_train_val_test_features
from library.config import TRAIN_META_PATH, VAL_META_PATH, TEST_META_PATH


def load_metadata():
    """
    Loads the metadata CSVs for train, validation, and test sets.

    Returns:
        tuple: (train_df, val_df, test_df) containing the metadata.
    """
    train_df = pd.read_csv(TRAIN_META_PATH)
    val_df = pd.read_csv(VAL_META_PATH)
    test_df = pd.read_csv(TEST_META_PATH)
    return train_df, val_df, test_df


def build_dataset(load_cached_data=True):
    """
    Constructs the full datasets for training, validation, and testing.
    Uses the feature engineering module to load or generate features.

    Args:
        load_cached_data (bool): If True, attempts to load pre-computed features
                                 from Parquet files. If False, forces regeneration.

    Returns:
        dict: A dictionary containing:
            'train': (X_train, y_train)
            'val': (X_val, y_val)
            'test': (X_test, test_ids)
    """
    # Retrieve feature dataframes (handles caching internally via feature_engineering.py)
    train_features_df, val_features_df, test_features_df = get_train_val_test_features(
        load_cached_data=load_cached_data
    )

    # --- Process Training Data ---
    # Drop segment_id as it is not a feature
    # Extract target variable
    if "time_to_eruption" in train_features_df.columns:
        y_train = train_features_df["time_to_eruption"]
        X_train = train_features_df.drop(columns=["segment_id", "time_to_eruption"])
    else:
        # Fallback if for some reason target is missing (should not happen for train)
        raise ValueError("Training data is missing 'time_to_eruption' column.")

    # --- Process Validation Data ---
    if "time_to_eruption" in val_features_df.columns:
        y_val = val_features_df["time_to_eruption"]
        X_val = val_features_df.drop(columns=["segment_id", "time_to_eruption"])
    else:
        raise ValueError("Validation data is missing 'time_to_eruption' column.")

    # --- Process Test Data ---
    # Test data does not have target. We need segment_id for submission.
    test_ids = test_features_df["segment_id"]
    X_test = test_features_df.drop(columns=["segment_id"])

    # Ensure no target column leaked into test features
    if "time_to_eruption" in X_test.columns:
        X_test = X_test.drop(columns=["time_to_eruption"])

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, test_ids),
    }
