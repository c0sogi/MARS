import os
import ast
import pandas as pd
import numpy as np
from library.config import Config


def get_feature_intersection(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list:
    """
    Identifies common columns between train and test datasets to prevent leakage.
    Excludes target, IDs, and known leakage columns.

    Args:
        train_df (pd.DataFrame): Training data.
        test_df (pd.DataFrame): Test data.

    Returns:
        list: List of safe feature column names sorted alphabetically.
    """
    # Identify common columns
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    common_cols = train_cols.intersection(test_cols)

    # Define columns to exclude (Targets, IDs, Leakage)
    # giver_username_if_known: Leakage (implies success)
    # source_file: Metadata artifact
    # request_id: ID
    # requester_received_pizza: Target
    exclude_cols = {
        "requester_received_pizza",
        "request_id",
        "giver_username_if_known",
        "source_file",
        "train_mask",
        "val_mask",
    }

    safe_features = [col for col in common_cols if col not in exclude_cols]
    safe_features.sort()

    return safe_features


def load_data(
    debug: bool = Config.DEBUG, load_cached_data: bool = Config.LOAD_CACHED_DATA
):
    """
    Loads training, validation, and test datasets.
    Performs basic cleaning and type conversion (e.g., parsing stringified lists).
    Implements caching using Parquet.

    Args:
        debug (bool): If True, loads a small subset of data.
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Define cache paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "train_cleaned.parquet")
    val_cache = os.path.join(cache_dir, "val_cleaned.parquet")
    test_cache = os.path.join(cache_dir, "test_cleaned.parquet")

    # 1. Try Loading from Cache
    # We only use cache if NOT in debug mode (to avoid caching/loading subsets)
    if load_cached_data and not debug:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            print("Loading data from cache...")
            try:
                # Use pyarrow engine to handle complex types if stored
                train_df = pd.read_parquet(train_cache)
                val_df = pd.read_parquet(val_cache)
                test_df = pd.read_parquet(test_cache)
                return train_df, val_df, test_df
            except Exception as e:
                print(f"Failed to load cache: {e}. Reloading from raw source.")

    # 2. Load Raw Data
    print("Loading raw data from metadata CSVs...")
    if not os.path.exists(Config.TRAIN_CSV):
        raise FileNotFoundError(f"Train CSV not found at {Config.TRAIN_CSV}")

    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # 3. Basic Cleaning & Type Conversion

    def parse_list_column(df, col_name):
        """Parses stringified list columns back to python lists."""
        if col_name in df.columns:
            # Check if column is object type and looks like a list string
            if df[col_name].dtype == "object":
                # Fill NaNs with empty list string representation
                df[col_name] = df[col_name].fillna("[]")
                # Apply literal_eval safely
                try:
                    df[col_name] = df[col_name].apply(
                        lambda x: (
                            ast.literal_eval(x)
                            if isinstance(x, str) and x.startswith("[")
                            else x
                        )
                    )
                except (ValueError, SyntaxError):
                    # Fallback if parsing fails, though data should be clean
                    pass
        return df

    # Columns known to be lists based on dataset description
    list_cols = ["requester_subreddits_at_request"]

    # Text columns to ensure are strings
    text_cols = ["request_text", "request_title", "request_text_edit_aware"]

    for df in [train_df, val_df, test_df]:
        # Parse lists
        for col in list_cols:
            parse_list_column(df, col)

        # Clean text
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)

        # Ensure target is boolean (only for train/val)
        if "requester_received_pizza" in df.columns:
            df["requester_received_pizza"] = df["requester_received_pizza"].astype(bool)

    # 4. Save to Cache (only if not debug)
    if not debug:
        print("Saving processed data to cache...")
        try:
            # Use pyarrow engine for better list support
            train_df.to_parquet(train_cache, index=False, engine="pyarrow")
            val_df.to_parquet(val_cache, index=False, engine="pyarrow")
            test_df.to_parquet(test_cache, index=False, engine="pyarrow")
        except Exception as e:
            print(f"Warning: Could not save to cache: {e}")

    # 5. Debug Sampling
    if debug:
        print(f"Debug mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    return train_df, val_df, test_df
