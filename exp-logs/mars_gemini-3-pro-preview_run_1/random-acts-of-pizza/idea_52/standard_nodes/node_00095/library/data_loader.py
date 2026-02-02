import os
import ast
import pandas as pd
import numpy as np
from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    WORKING_DIR,
    DEBUG_SAMPLE_SIZE,
    RANDOM_STATE,
)
from library.utils import set_seed


def load_data(load_cached_data=True, debug=False):
    """
    Loads the training, validation, and test datasets.

    Implements a caching mechanism using Parquet files to speed up subsequent loads.
    Handles initial formatting such as parsing stringified lists and ensuring correct data types.

    Args:
        load_cached_data (bool): If True, attempts to load from the local cache first.
        debug (bool): If True, downsamples the datasets to DEBUG_SAMPLE_SIZE for rapid testing.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    set_seed(RANDOM_STATE)

    # Define cache paths
    train_cache = os.path.join(WORKING_DIR, "train_processed.parquet")
    val_cache = os.path.join(WORKING_DIR, "val_processed.parquet")
    test_cache = os.path.join(WORKING_DIR, "test_processed.parquet")

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    data_loaded = False
    train_df, val_df, test_df = None, None, None

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            try:
                print("Loading data from cache...")
                train_df = pd.read_parquet(train_cache)
                val_df = pd.read_parquet(val_cache)
                test_df = pd.read_parquet(test_cache)
                data_loaded = True
            except Exception as e:
                print(f"Failed to load cache: {e}. Reloading from source.")

    # 2. If not loaded or cache disabled, process from metadata CSVs
    if not data_loaded:
        print("Loading data from metadata CSVs...")

        # Load CSVs
        if not os.path.exists(TRAIN_PATH):
            raise FileNotFoundError(f"Train metadata file not found at {TRAIN_PATH}")
        if not os.path.exists(VAL_PATH):
            raise FileNotFoundError(f"Validation metadata file not found at {VAL_PATH}")
        if not os.path.exists(TEST_PATH):
            raise FileNotFoundError(f"Test metadata file not found at {TEST_PATH}")

        train_df = pd.read_csv(TRAIN_PATH)
        val_df = pd.read_csv(VAL_PATH)
        test_df = pd.read_csv(TEST_PATH)

        # --- Processing ---

        # 1. Parse 'requester_subreddits_at_request' which is a stringified list in CSV
        # e.g., "['sub1', 'sub2']" -> ['sub1', 'sub2']
        list_col = "requester_subreddits_at_request"

        for df in [train_df, val_df, test_df]:
            if list_col in df.columns:
                # Handle NaNs by converting to empty list string first if necessary,
                # though metadata shouldn't have NaNs for this field based on schema.
                # using literal_eval safely
                df[list_col] = (
                    df[list_col]
                    .fillna("[]")
                    .apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                )

        # 2. Ensure target column is integer (0/1) for Train/Val
        target_col = "requester_received_pizza"
        for df in [train_df, val_df]:
            if target_col in df.columns:
                df[target_col] = df[target_col].astype(int)

        # 3. Ensure 'post_was_edited' is boolean or consistent
        # In CSV it might be "False" (str) or False (bool)
        edit_col = "post_was_edited"
        for df in [train_df, val_df, test_df]:
            if edit_col in df.columns:
                # Convert to boolean, handling potential string representations
                df[edit_col] = df[edit_col].replace({"False": False, "True": True})
                df[edit_col] = df[edit_col].astype(bool)

        # 4. Save to cache (Parquet supports lists via pyarrow engine)
        print("Saving processed data to cache...")
        try:
            train_df.to_parquet(train_cache, engine="pyarrow", index=False)
            val_df.to_parquet(val_cache, engine="pyarrow", index=False)
            test_df.to_parquet(test_cache, engine="pyarrow", index=False)
        except Exception as e:
            print(f"Warning: Could not save to parquet cache. Error: {e}")

    # 3. Handle Debugging (Downsampling)
    if debug:
        print(f"Debug mode enabled. Downsampling to {DEBUG_SAMPLE_SIZE} samples.")
        train_df = train_df.head(DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(DEBUG_SAMPLE_SIZE)

    print(
        f"Data loaded. Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}"
    )

    return train_df, val_df, test_df
