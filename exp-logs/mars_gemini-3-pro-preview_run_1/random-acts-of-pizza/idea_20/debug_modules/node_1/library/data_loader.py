import os
import ast
import pandas as pd
import library.config as config


def parse_list_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Parses columns containing stringified lists into actual Python lists.

    Args:
        df: Input DataFrame.
        columns: List of column names to parse.

    Returns:
        DataFrame with parsed columns.
    """
    df_out = df.copy()
    for col in columns:
        if col in df_out.columns:
            # Use ast.literal_eval to safely parse string representation of lists
            # We handle potential NaNs by treating them as empty lists or keeping them as is
            df_out[col] = df_out[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
    return df_out


def load_dataset(load_cached_data: bool = True):
    """
    Loads the train, validation, and test datasets.
    Handles caching to parquet files to speed up subsequent loads.
    Parses list columns like 'requester_subreddits_at_request'.
    Respects the config.DEBUG flag by slicing data if set.

    Args:
        load_cached_data: If True, attempts to load from parquet cache.

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Define cache paths
    train_cache = os.path.join(config.WORKING_DIR, "train_parsed.parquet")
    val_cache = os.path.join(config.WORKING_DIR, "val_parsed.parquet")
    test_cache = os.path.join(config.WORKING_DIR, "test_parsed.parquet")

    # Columns that need parsing from string to list
    list_cols = ["requester_subreddits_at_request"]

    # Check if we can load from cache
    if (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    ):
        print("Loading datasets from cache...")
        train_df = pd.read_parquet(train_cache)
        val_df = pd.read_parquet(val_cache)
        test_df = pd.read_parquet(test_cache)
    else:
        print("Loading datasets from raw CSVs...")
        # Load raw CSVs
        train_df = pd.read_csv(config.TRAIN_PATH)
        val_df = pd.read_csv(config.VAL_PATH)
        test_df = pd.read_csv(config.TEST_PATH)

        # Parse list columns
        print("Parsing list columns...")
        train_df = parse_list_columns(train_df, list_cols)
        val_df = parse_list_columns(val_df, list_cols)
        test_df = parse_list_columns(test_df, list_cols)

        # Save to cache (full datasets)
        print(f"Saving processed datasets to {config.WORKING_DIR}...")
        train_df.to_parquet(train_cache, index=False)
        val_df.to_parquet(val_cache, index=False)
        test_df.to_parquet(test_cache, index=False)

    # Handle DEBUG mode
    if config.DEBUG:
        print("DEBUG mode enabled: Slicing datasets to 50 rows.")
        train_df = train_df.head(50)
        val_df = val_df.head(50)
        test_df = test_df.head(50)

    return train_df, val_df, test_df
