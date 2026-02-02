import pandas as pd
import ast
import os
from library.config import Config


def load_datasets():
    """
    Loads the train, validation, and test datasets from the paths defined in Config.
    Parses stringified list columns back into Python lists.
    Applies debug sampling if Config.DEBUG is True.

    Returns:
        tuple: (df_train, df_val, df_test)
    """
    print(f"Loading datasets from {Config.METADATA_DIR}...")

    # Load CSV files
    df_train = pd.read_csv(Config.TRAIN_PATH)
    df_val = pd.read_csv(Config.VAL_PATH)
    df_test = pd.read_csv(Config.TEST_PATH)

    # Parse stringified list columns
    # The subreddit column is stored as a string representation of a list in the CSV
    # e.g., "['sub1', 'sub2']". We convert it back to a list object.
    print(f"Parsing {Config.SUBREDDIT_COL} column...")

    def parse_list_col(val):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []

    if Config.SUBREDDIT_COL in df_train.columns:
        df_train[Config.SUBREDDIT_COL] = df_train[Config.SUBREDDIT_COL].apply(
            parse_list_col
        )

    if Config.SUBREDDIT_COL in df_val.columns:
        df_val[Config.SUBREDDIT_COL] = df_val[Config.SUBREDDIT_COL].apply(
            parse_list_col
        )

    if Config.SUBREDDIT_COL in df_test.columns:
        df_test[Config.SUBREDDIT_COL] = df_test[Config.SUBREDDIT_COL].apply(
            parse_list_col
        )

    # Apply Debug Sampling
    if Config.DEBUG:
        print(
            f"DEBUG mode enabled. Sampling {Config.DEBUG_SAMPLE_SIZE} rows per dataset."
        )
        df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)
        df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)
        df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

    print(
        f"Data loaded. Train: {df_train.shape}, Val: {df_val.shape}, Test: {df_test.shape}"
    )

    return df_train, df_val, df_test
