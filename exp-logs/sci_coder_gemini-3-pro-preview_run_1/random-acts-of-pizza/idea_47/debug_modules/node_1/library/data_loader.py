import os
import ast
import pandas as pd
from library.config import TRAIN_PATH, VAL_PATH, TEST_PATH, WORKING_DIR, TARGET_COL
from library.utils import seed_everything


def load_dataset(load_cached_data: bool = True):
    """
    Loads the train, validation, and test datasets.

    This function handles:
    1. Caching: Checks for pre-processed parquet files to save time.
    2. Loading: Reads from metadata CSVs if cache is not available.
    3. Parsing: Converts stringified list columns (e.g. subreddits) back to lists.
    4. Saving: Caches the processed dataframes to parquet.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache first.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    seed_everything()

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Define cache paths
    train_cache = os.path.join(WORKING_DIR, "train_base.parquet")
    val_cache = os.path.join(WORKING_DIR, "val_base.parquet")
    test_cache = os.path.join(WORKING_DIR, "test_base.parquet")

    # Attempt to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            try:
                print("Loading datasets from cache...")
                train_df = pd.read_parquet(train_cache)
                val_df = pd.read_parquet(val_cache)
                test_df = pd.read_parquet(test_cache)
                return train_df, val_df, test_df
            except Exception as e:
                print(f"Failed to load cache: {e}. Reloading from source.")

    print("Loading datasets from source CSVs...")

    # Load CSVs
    # Converters are used to parse the stringified list representation in CSV back to python lists
    # 'requester_subreddits_at_request' is stored as "['sub1', 'sub2']" in CSV
    converters = {
        "requester_subreddits_at_request": lambda x: (
            ast.literal_eval(x) if isinstance(x, str) else []
        )
    }

    train_df = pd.read_csv(TRAIN_PATH, converters=converters)
    val_df = pd.read_csv(VAL_PATH, converters=converters)
    test_df = pd.read_csv(TEST_PATH, converters=converters)

    # Ensure target column is properly typed (boolean or int)
    if TARGET_COL in train_df.columns:
        train_df[TARGET_COL] = train_df[TARGET_COL].astype(int)
    if TARGET_COL in val_df.columns:
        val_df[TARGET_COL] = val_df[TARGET_COL].astype(int)

    # Save to cache
    print("Saving datasets to cache...")
    train_df.to_parquet(train_cache, index=False)
    val_df.to_parquet(val_cache, index=False)
    test_df.to_parquet(test_cache, index=False)

    return train_df, val_df, test_df
