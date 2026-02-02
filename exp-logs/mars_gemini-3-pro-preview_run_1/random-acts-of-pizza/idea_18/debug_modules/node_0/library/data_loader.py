import os
import pandas as pd
from library.config import Config
from library.utils import parse_stringified_list


def load_data(load_cached_data=True):
    """
    Loads the train, validation, and test datasets.

    If cached Parquet files exist in the working directory and load_cached_data is True,
    it loads from there to save processing time.

    Otherwise, it reads from the metadata CSVs, parses the stringified 'requester_subreddits_at_request'
    column into actual Python lists, saves the result to Parquet for caching, and returns the DataFrames.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache. Defaults to True.

    Returns:
        tuple: (df_train, df_val, df_test)
    """

    # Define cache file paths
    train_cache_path = os.path.join(Config.WORKING_DIR, "train_cleaned.parquet")
    val_cache_path = os.path.join(Config.WORKING_DIR, "val_cleaned.parquet")
    test_cache_path = os.path.join(Config.WORKING_DIR, "test_cleaned.parquet")

    # 1. Try loading from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache_path)
            and os.path.exists(val_cache_path)
            and os.path.exists(test_cache_path)
        ):
            print("Loading datasets from cache...")
            df_train = pd.read_parquet(train_cache_path)
            df_val = pd.read_parquet(val_cache_path)
            df_test = pd.read_parquet(test_cache_path)
            return df_train, df_val, df_test
        else:
            print("Cache not found. Loading from raw metadata...")
    else:
        print("Ignoring cache. Loading from raw metadata...")

    # 2. Load from raw CSVs
    if not os.path.exists(Config.TRAIN_PATH):
        raise FileNotFoundError(f"Train file not found at {Config.TRAIN_PATH}")
    if not os.path.exists(Config.VAL_PATH):
        raise FileNotFoundError(f"Validation file not found at {Config.VAL_PATH}")
    if not os.path.exists(Config.TEST_PATH):
        raise FileNotFoundError(f"Test file not found at {Config.TEST_PATH}")

    df_train = pd.read_csv(Config.TRAIN_PATH)
    df_val = pd.read_csv(Config.VAL_PATH)
    df_test = pd.read_csv(Config.TEST_PATH)

    # 3. Process columns
    # The 'requester_subreddits_at_request' column is stored as a string representation of a list in CSV.
    # We parse it back to a Python list.
    target_col = "requester_subreddits_at_request"

    print(f"Parsing '{target_col}' column...")

    # Apply parsing logic
    # We use the utility function imported from library.utils
    if target_col in df_train.columns:
        df_train[target_col] = df_train[target_col].apply(parse_stringified_list)

    if target_col in df_val.columns:
        df_val[target_col] = df_val[target_col].apply(parse_stringified_list)

    if target_col in df_test.columns:
        df_test[target_col] = df_test[target_col].apply(parse_stringified_list)

    # 4. Save to cache
    print("Saving processed datasets to cache...")
    # Ensure working directory exists (handled by Config, but good practice)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    df_train.to_parquet(train_cache_path, index=False)
    df_val.to_parquet(val_cache_path, index=False)
    df_test.to_parquet(test_cache_path, index=False)

    return df_train, df_val, df_test
