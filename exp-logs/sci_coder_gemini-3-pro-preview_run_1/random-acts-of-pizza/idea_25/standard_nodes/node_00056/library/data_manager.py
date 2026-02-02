import os
import pandas as pd
from library.utils import load_metadata, get_feature_intersection, Timer

# Define cache directory
CACHE_DIR = "./working/idea_25/"


def load_dataset(split):
    """
    Loads the metadata CSV for a specific split (train, val, test).
    Wrapper around library.utils.load_metadata.

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    return load_metadata(split)


def get_clean_data(load_cached_data=True, debug_mode=False, debug_size=50):
    """
    Loads train, val, and test datasets, ensures feature consistency between them
    (leakage prevention), and returns the cleaned dataframes.

    Implements caching for the processed datasets to optimize runtime.

    Args:
        load_cached_data (bool): Whether to load from cache if available.
        debug_mode (bool): If True, returns a small subset of the data for debugging.
        debug_size (int): Number of samples to return in debug mode.

    Returns:
        tuple: (df_train, df_val, df_test)
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    train_cache = os.path.join(CACHE_DIR, "train_clean.parquet")
    val_cache = os.path.join(CACHE_DIR, "val_clean.parquet")
    test_cache = os.path.join(CACHE_DIR, "test_clean.parquet")

    # 1. Try loading from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            print(f"Loading cleaned data from cache ({CACHE_DIR})...")
            try:
                df_train = pd.read_parquet(train_cache)
                df_val = pd.read_parquet(val_cache)
                df_test = pd.read_parquet(test_cache)

                if debug_mode:
                    print(f"Debug mode: Subsampling to {debug_size} samples.")
                    return (
                        df_train.head(debug_size),
                        df_val.head(debug_size),
                        df_test.head(debug_size),
                    )

                return df_train, df_val, df_test
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute (Load raw -> Clean -> Save)
    print("Computing cleaned data from scratch...")
    with Timer("Data Cleaning"):
        # Load raw metadata
        df_train_raw = load_dataset("train")
        df_val_raw = load_dataset("val")
        df_test_raw = load_dataset("test")

        # Identify intersection of features to prevent leakage
        # We exclude the target from the intersection check, but add it back to train/val
        target_col = "requester_received_pizza"
        exclude_cols = [target_col]

        # get_feature_intersection returns sorted list of shared columns
        common_features = get_feature_intersection(
            df_train_raw, df_test_raw, exclude_cols=exclude_cols
        )

        # Define columns to keep
        # Ensure request_id is kept (it is typically in common_features if present in both)
        cols_train = common_features + [target_col]
        cols_test = common_features

        # Filter dataframes
        df_train = df_train_raw[cols_train].copy()
        df_val = df_val_raw[cols_train].copy()
        df_test = df_test_raw[cols_test].copy()

        # Save to cache
        print(f"Saving cleaned data to {CACHE_DIR}...")
        df_train.to_parquet(train_cache, index=False)
        df_val.to_parquet(val_cache, index=False)
        df_test.to_parquet(test_cache, index=False)

    # 3. Return (with optional debug slicing)
    if debug_mode:
        print(f"Debug mode: Subsampling to {debug_size} samples.")
        return (
            df_train.head(debug_size),
            df_val.head(debug_size),
            df_test.head(debug_size),
        )

    return df_train, df_val, df_test
