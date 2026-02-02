import os
import pandas as pd
import library.config as config
import library.features as features


def build_dataset(split="train", load_cached_data=True):
    """
    Builds the dataset for the requested split ('train', 'val', 'test').
    Leverages the caching and processing logic in library.features to load
    raw data, extract features, and aggregate them into a DataFrame.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to try loading from cache first.

    Returns:
        pd.DataFrame: The processed dataset containing features, segment_id,
                      and time_to_eruption (for train/val).
    """
    # Determine metadata path and cache filename based on the requested split
    if split == "train":
        metadata_path = config.TRAIN_METADATA_PATH
        save_name = "train_features"
    elif split == "val":
        metadata_path = config.VAL_METADATA_PATH
        save_name = "val_features"
    elif split == "test":
        metadata_path = config.TEST_METADATA_PATH
        save_name = "test_features"
    else:
        raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'.")

    # Delegate the heavy lifting to the library function which handles:
    # 1. Caching logic (checking config.WORKING_DIR for parquet files)
    # 2. Iterating through metadata
    # 3. Loading raw CSVs and extracting features via features.extract_segment_features
    # 4. Saving computed results to cache
    df = features.process_dataset(
        metadata_path=metadata_path,
        load_cached_data=load_cached_data,
        save_name=save_name,
    )

    return df


def prepare_features_target(df, target_col="time_to_eruption", id_col="segment_id"):
    """
    Splits the processed dataframe into feature matrix X and target vector y.
    Removes the identifier column from X to ensure the model only trains on signal features.

    Args:
        df (pd.DataFrame): The dataframe returned by build_dataset.
        target_col (str): Name of the target column.
        id_col (str): Name of the ID column to exclude from features.

    Returns:
        tuple: (X, y) where X is a DataFrame of features and y is a Series (or None if target missing).
    """
    # Identify columns to drop (ID and Target) to isolate features
    drop_cols = [id_col]

    if target_col in df.columns:
        drop_cols.append(target_col)
        y = df[target_col]
    else:
        y = None

    # Ensure we only attempt to drop columns that actually exist in the DataFrame
    existing_drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=existing_drop_cols)

    return X, y
