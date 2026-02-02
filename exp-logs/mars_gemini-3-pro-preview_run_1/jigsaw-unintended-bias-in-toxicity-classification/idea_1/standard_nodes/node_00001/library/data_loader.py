import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, JigsawMetrics


def bias_resampling(df, weight=Config.RESAMPLE_WEIGHT):
    """
    Performs stratified oversampling on examples that mention identities.

    This function identifies rows where any of the tracked identities are present
    (value >= 0.5) and duplicates them 'weight' times. This helps the model
    learn to distinguish between toxic and non-toxic uses of identity terms
    (improving BPSN and BNSP metrics).

    Args:
        df (pd.DataFrame): The training dataframe.
        weight (int): Number of times to duplicate the identity-mentioning rows.

    Returns:
        pd.DataFrame: The resampled dataframe, shuffled.
    """
    # Get list of identities from JigsawMetrics
    metrics_helper = JigsawMetrics()
    identities = metrics_helper.identities

    # Create a mask for rows that mention ANY identity
    # We fill NaNs with 0.0 to assume no mention if data is missing
    identity_mask = pd.Series(False, index=df.index)

    for identity in identities:
        if identity in df.columns:
            # Standard threshold for "mention" is >= 0.5
            col_mask = df[identity].fillna(0.0) >= 0.5
            identity_mask = identity_mask | col_mask

    # Filter rows to duplicate
    rows_to_duplicate = df[identity_mask]

    if len(rows_to_duplicate) == 0:
        return df

    # Concatenate original data with duplicated rows
    # We duplicate the subset 'weight' times
    dfs_to_concat = [df] + [rows_to_duplicate] * weight
    resampled_df = pd.concat(dfs_to_concat, axis=0, ignore_index=True)

    # Shuffle the dataset to break order
    resampled_df = resampled_df.sample(frac=1, random_state=Config.SEED).reset_index(
        drop=True
    )

    return resampled_df


def load_data(load_cached_data=True, debug=Config.DEBUG):
    """
    Loads the Train, Validation, and Test datasets. Handles caching and bias resampling.

    Args:
        load_cached_data (bool): If True, attempts to load processed training data from Parquet cache.
        debug (bool): If True, loads a small subset of the data for debugging and skips caching.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    set_seed(Config.SEED)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    train_cache_path = os.path.join(Config.WORKING_DIR, "train_resampled.parquet")

    # ---------------------------------------------------------
    # Scenario A: Load from Cache (Not Debug)
    # ---------------------------------------------------------
    if load_cached_data and not debug and os.path.exists(train_cache_path):
        print(f"Loading resampled training data from cache: {train_cache_path}")
        try:
            train_df = pd.read_parquet(train_cache_path)

            # Load Val and Test from raw metadata (fast enough to not need caching)
            val_df = pd.read_csv(Config.VAL_DATA_PATH)
            test_df = pd.read_csv(Config.TEST_DATA_PATH)

            # Ensure text is string and handle NaNs
            val_df["comment_text"] = val_df["comment_text"].fillna("")
            test_df["comment_text"] = test_df["comment_text"].fillna("")

            return train_df, val_df, test_df
        except Exception as e:
            print(f"Failed to load cache: {e}. Falling back to raw data processing.")

    # ---------------------------------------------------------
    # Scenario B: Process from Scratch
    # ---------------------------------------------------------
    print("Loading raw data from metadata...")
    train_df = pd.read_csv(Config.TRAIN_DATA_PATH)
    val_df = pd.read_csv(Config.VAL_DATA_PATH)
    test_df = pd.read_csv(Config.TEST_DATA_PATH)

    # Handle missing text
    train_df["comment_text"] = train_df["comment_text"].fillna("")
    val_df["comment_text"] = val_df["comment_text"].fillna("")
    test_df["comment_text"] = test_df["comment_text"].fillna("")

    # ---------------------------------------------------------
    # Debug Subsampling
    # ---------------------------------------------------------
    if debug:
        print(f"DEBUG MODE: Subsampling data to {Config.DEBUG_SAMPLE_SIZE} rows.")
        train_df = train_df.sample(
            n=min(len(train_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        test_df = test_df.sample(
            n=min(len(test_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Bias Correction (Resampling)
    # ---------------------------------------------------------
    print(f"Applying Bias-Corrective Resampling (Weight={Config.RESAMPLE_WEIGHT})...")
    print(f"Original Train Shape: {train_df.shape}")

    train_df = bias_resampling(train_df, weight=Config.RESAMPLE_WEIGHT)

    print(f"Resampled Train Shape: {train_df.shape}")

    # ---------------------------------------------------------
    # Save to Cache (If not debug)
    # ---------------------------------------------------------
    if not debug:
        print(f"Saving resampled training data to cache: {train_cache_path}")
        train_df.to_parquet(train_cache_path)

    return train_df, val_df, test_df
