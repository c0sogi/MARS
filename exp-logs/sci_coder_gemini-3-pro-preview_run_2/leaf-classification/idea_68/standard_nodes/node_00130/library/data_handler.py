import os
import numpy as np
import pandas as pd
from library.config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    CACHE_DIR,
    MARGIN_COLS,
    SHAPE_COLS,
    TEXTURE_COLS,
    ALL_FEATURE_COLS,
    DTYPE,
)
from library.image_features import get_morphometric_features
from library.utils import set_seed

# Define Morphometric Column Names based on image_features.py output
MORPH_COLS = [f"hu_{i}" for i in range(7)] + [
    "aspect_ratio",
    "solidity",
    "extent",
    "eccentricity",
]


def load_merged_data(load_cached_data=True):
    """
    Loads metadata and image features, merges them, and handles caching.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (df_train, df_val, df_test) merged DataFrames.
    """
    # Define cache paths
    train_cache = os.path.join(CACHE_DIR, "train_merged.parquet")
    val_cache = os.path.join(CACHE_DIR, "val_merged.parquet")
    test_cache = os.path.join(CACHE_DIR, "test_merged.parquet")

    # Check if cache exists
    if (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    ):
        print("Loading merged data from cache...")
        df_train = pd.read_parquet(train_cache)
        df_val = pd.read_parquet(val_cache)
        df_test = pd.read_parquet(test_cache)
        return df_train, df_val, df_test

    print("Generating merged data from scratch...")

    # 1. Load Metadata (Tabular Features + Labels)
    df_train_meta = pd.read_csv(TRAIN_DATA_PATH)
    df_val_meta = pd.read_csv(VAL_DATA_PATH)
    df_test_meta = pd.read_csv(TEST_DATA_PATH)

    # 2. Load Image Features (Morphometrics)
    # image_features.py handles its own caching for the extraction part
    df_train_img, df_val_img, df_test_img = get_morphometric_features(
        load_cached_data=load_cached_data
    )

    # 3. Merge DataFrames on 'id'
    # Inner join ensures we only keep records that exist in both (should be all)
    df_train = pd.merge(df_train_meta, df_train_img, on="id", how="inner")
    df_val = pd.merge(df_val_meta, df_val_img, on="id", how="inner")
    df_test = pd.merge(df_test_meta, df_test_img, on="id", how="inner")

    # 4. Save to Cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    df_train.to_parquet(train_cache, index=False)
    df_val.to_parquet(val_cache, index=False)
    df_test.to_parquet(test_cache, index=False)

    return df_train, df_val, df_test


def get_feature_groups(df):
    """
    Splits a DataFrame into the specific feature views required by the ensemble.
    Casts all data to float64.

    Args:
        df (pd.DataFrame): The dataframe containing all features.

    Returns:
        dict: A dictionary of feature matrices (np.ndarray).
    """
    # Extract specific views
    # 1. Global View (All original tabular features)
    X_global = df[ALL_FEATURE_COLS].values.astype(DTYPE)

    # 2. Stratified Views
    X_margin = df[MARGIN_COLS].values.astype(DTYPE)
    X_shape = df[SHAPE_COLS].values.astype(DTYPE)
    X_texture = df[TEXTURE_COLS].values.astype(DTYPE)

    # 3. Morphometric View (Extracted from images)
    # Ensure columns exist, fill with 0 if missing (though they should exist)
    missing_morph = [c for c in MORPH_COLS if c not in df.columns]
    if missing_morph:
        print(
            f"Warning: Missing morphometric columns: {missing_morph}. Filling with 0."
        )
        for c in missing_morph:
            df[c] = 0.0

    X_morph = df[MORPH_COLS].values.astype(DTYPE)

    return {
        "Global": X_global,
        "Margin": X_margin,
        "Shape": X_shape,
        "Texture": X_texture,
        "Morphometrics": X_morph,
    }


def get_data(load_cached_data=True):
    """
    Main entry point to retrieve prepared data for training and inference.

    Args:
        load_cached_data (bool): Whether to use cached data.

    Returns:
        tuple: (X_train_dict, y_train, X_val_dict, y_val, X_test_dict, test_ids, classes)
    """
    set_seed()

    # Load Merged Data
    df_train, df_val, df_test = load_merged_data(load_cached_data)

    # Extract Labels (Species)
    y_train = df_train["species"].values
    y_val = df_val["species"].values

    # Extract Test IDs
    test_ids = df_test["id"].values

    # Get Unique Classes (Sorted Alphabetically for consistency)
    classes = sorted(np.unique(y_train))

    # Generate Feature Views
    print("Generating feature views...")
    X_train_dict = get_feature_groups(df_train)
    X_val_dict = get_feature_groups(df_val)
    X_test_dict = get_feature_groups(df_test)

    print(f"Data Loaded:")
    print(f"  Train: {len(df_train)} samples")
    print(f"  Val:   {len(df_val)} samples")
    print(f"  Test:  {len(df_test)} samples")
    print(f"  Classes: {len(classes)}")

    return X_train_dict, y_train, X_val_dict, y_val, X_test_dict, test_ids, classes
