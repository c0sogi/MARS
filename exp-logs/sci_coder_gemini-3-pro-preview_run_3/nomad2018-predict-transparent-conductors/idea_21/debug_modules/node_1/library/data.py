import os
import numpy as np
import pandas as pd
from library.features import generate_features
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TRAIN_FEATURES_PATH,
    VAL_FEATURES_PATH,
    TEST_FEATURES_PATH,
    TARGET_COLS,
    RANDOM_SEED,
)


def process_geometry_data(metadata_path, output_path, load_cached_data=True):
    """
    Orchestrates the processing of geometry data by calling the feature extraction
    pipeline. This function handles the loading of metadata, iteration over samples,
    extraction of geometric features (Physical, RDF, DLE), and merging with tabular data.
    It relies on the caching mechanism implemented in library.features.generate_features.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        output_path (str): Path where the processed features parquet file will be saved/loaded.
        load_cached_data (bool): If True, attempts to load from output_path first.

    Returns:
        pd.DataFrame: DataFrame containing the processed features and IDs.
    """
    # Delegate to the library function which implements the specific extraction and caching logic
    return generate_features(
        metadata_path, output_path, load_cached_data=load_cached_data
    )


def prepare_train_test_data(load_cached_data=True):
    """
    Loads the processed feature sets for training, validation, and testing.
    Aligns features with targets, removes constant columns based on training data statistics,
    and applies logarithmic transformation to the targets.

    Args:
        load_cached_data (bool): Whether to use cached feature files.

    Returns:
        tuple: (X_train, y_train_dict, X_val, y_val_dict, X_test, test_ids)
            - X_train, X_val, X_test: Feature DataFrames.
            - y_train_dict, y_val_dict: Dictionaries mapping target names to log-transformed target Series.
            - test_ids: Array of IDs for the test set.
    """
    print("Preparing train/val/test data...")

    # 1. Generate or Load Features
    # The generate_features function handles the heavy lifting of reading XYZ files
    # and computing descriptors. It returns a DF with features + tabular metadata + ID.
    df_train_feats = process_geometry_data(
        TRAIN_METADATA_PATH, TRAIN_FEATURES_PATH, load_cached_data
    )
    df_val_feats = process_geometry_data(
        VAL_METADATA_PATH, VAL_FEATURES_PATH, load_cached_data
    )
    df_test_feats = process_geometry_data(
        TEST_METADATA_PATH, TEST_FEATURES_PATH, load_cached_data
    )

    # 2. Load Metadata to retrieve targets
    # generate_features explicitly excludes target columns, so we fetch them here.
    df_train_meta = pd.read_csv(TRAIN_METADATA_PATH)
    df_val_meta = pd.read_csv(VAL_METADATA_PATH)
    df_test_meta = pd.read_csv(TEST_METADATA_PATH)

    # 3. Merge Features with Targets using 'id'
    # This ensures perfect alignment between the computed features and the ground truth.
    train_merged = pd.merge(
        df_train_feats, df_train_meta[["id"] + TARGET_COLS], on="id", how="left"
    )
    val_merged = pd.merge(
        df_val_feats, df_val_meta[["id"] + TARGET_COLS], on="id", how="left"
    )
    test_merged = pd.merge(df_test_feats, df_test_meta[["id"]], on="id", how="left")

    # 4. Separate Features and Targets
    # Define feature columns as everything except 'id' and the target columns
    # Note: 'id' is preserved in test_ids for submission mapping
    exclude_cols = ["id"] + TARGET_COLS
    feature_cols = [c for c in train_merged.columns if c not in exclude_cols]

    X_train = train_merged[feature_cols]
    X_val = val_merged[feature_cols]
    X_test = test_merged[feature_cols]

    # 5. Feature Cleaning: Drop Constant Columns
    # We calculate standard deviation on the training set. If std == 0, the feature is constant.
    print("Dropping constant columns...")
    std_series = X_train.std()
    constant_cols = std_series[std_series == 0].index.tolist()

    if constant_cols:
        print(
            f"Found {len(constant_cols)} constant columns. Dropping them from all sets."
        )
        X_train = X_train.drop(columns=constant_cols)
        X_val = X_val.drop(columns=constant_cols)
        X_test = X_test.drop(columns=constant_cols)
    else:
        print("No constant columns found.")

    # 6. Prepare Targets with Log Transformation
    # Apply natural log transformation: z = log(1 + y)
    y_train_dict = {}
    y_val_dict = {}

    for target in TARGET_COLS:
        y_train_dict[target] = np.log1p(train_merged[target])
        y_val_dict[target] = np.log1p(val_merged[target])

    test_ids = test_merged["id"].values

    print(f"Data preparation complete.")
    print(f"Train feature shape: {X_train.shape}")
    print(f"Val feature shape:   {X_val.shape}")
    print(f"Test feature shape:  {X_test.shape}")

    return X_train, y_train_dict, X_val, y_val_dict, X_test, test_ids
