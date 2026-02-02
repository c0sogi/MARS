import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import Timer, set_seed


def load_datasets():
    """
    Loads the raw stratified datasets from the metadata directory.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    with Timer("Loading Metadata Parquet Files"):
        train_df = pd.read_parquet(Config.TRAIN_PATH)
        val_df = pd.read_parquet(Config.VAL_PATH)
        test_df = pd.read_parquet(Config.TEST_PATH)
    return train_df, val_df, test_df


def construct_unified_document(df):
    """
    Constructs the unified text document by concatenating Title, Body, and Prefixed History.
    Also generates the specialized views: Lexical (Title+Body) and Community (History).

    Args:
        df (pd.DataFrame): Input dataframe containing text and subreddit columns.

    Returns:
        pd.DataFrame: DataFrame with added columns 'text_unified', 'text_lexical', 'text_community'.
    """

    # Helper to ensure string
    def clean_str(s):
        return str(s) if s is not None else ""

    # Helper to process subreddits
    def process_subreddits(sub_list):
        if isinstance(sub_list, np.ndarray):
            sub_list = sub_list.tolist()
        if not isinstance(sub_list, list):
            return ""
        # Apply prefix 'sub_' to each subreddit to avoid namespace collisions
        # e.g., 'AskReddit' -> 'sub_AskReddit'
        return " ".join([f"sub_{str(s).strip()}" for s in sub_list])

    # Extract components
    titles = df[Config.TEXT_COLS[0]].apply(clean_str)
    bodies = df[Config.TEXT_COLS[1]].apply(clean_str)

    # Process community history
    # Note: requester_subreddits_at_request is expected to be a list/array column
    histories = df[Config.SUBREDDIT_COL].apply(process_subreddits)

    # 1. Lexical View: Title + Body
    text_lexical = titles + " " + bodies

    # 2. Community View: Prefixed History
    text_community = histories

    # 3. Unified View: Title + Body + Prefixed History
    text_unified = text_lexical + " " + text_community

    # Return as new columns
    out_df = df.copy()
    out_df["text_lexical"] = text_lexical
    out_df["text_community"] = text_community
    out_df["text_unified"] = text_unified

    return out_df


def extract_metadata(df):
    """
    Extracts the allow-listed numerical metadata columns.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: DataFrame containing only the selected metadata columns.
    """
    # Select columns defined in Config
    meta_cols = [c for c in Config.METADATA_COLS if c in df.columns]
    return df[meta_cols].copy()


def get_processed_data(load_cached_data=True):
    """
    Orchestrates the data loading, cleaning, and feature construction pipeline.
    Implements caching to disk to speed up subsequent runs.

    Args:
        load_cached_data (bool): If True, attempts to load from disk cache.

    Returns:
        tuple: (train_df, val_df, test_df) with all feature columns prepared.
    """
    # Cache paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "train_processed.parquet")
    val_cache = os.path.join(cache_dir, "val_processed.parquet")
    test_cache = os.path.join(cache_dir, "test_processed.parquet")

    # 1. Try Loading from Cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            print(f"Loading processed data from cache: {cache_dir}")
            try:
                train_df = pd.read_parquet(train_cache)
                val_df = pd.read_parquet(val_cache)
                test_df = pd.read_parquet(test_cache)
                return train_df, val_df, test_df
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")
        else:
            print("Cache not found. Processing from scratch...")
    else:
        print("Ignoring cache. Processing from scratch...")

    # 2. Load Raw Data
    train_df, val_df, test_df = load_datasets()

    # 3. Text Feature Construction (Unified, Lexical, Community)
    with Timer("Constructing Text Views"):
        train_df = construct_unified_document(train_df)
        val_df = construct_unified_document(val_df)
        test_df = construct_unified_document(test_df)

    # 4. Metadata Extraction & Imputation
    with Timer("Processing Metadata (Imputation)"):
        # Extract raw metadata subsets
        train_meta = extract_metadata(train_df)
        val_meta = extract_metadata(val_df)
        test_meta = extract_metadata(test_df)

        # Calculate Median on TRAIN set only (Leakage Prevention)
        impute_values = train_meta.median()

        # Apply imputation
        train_meta = train_meta.fillna(impute_values)
        val_meta = val_meta.fillna(impute_values)
        test_meta = test_meta.fillna(impute_values)

        # Update the main dataframes with cleaned metadata
        # We replace the original columns with the imputed ones
        for col in train_meta.columns:
            train_df[col] = train_meta[col]
            val_df[col] = val_meta[col]
            test_df[col] = test_meta[col]

    # 5. Save to Cache
    with Timer("Saving to Cache"):
        # Parquet handles strings and floats efficiently.
        # Note: We drop the original list column 'requester_subreddits_at_request'
        # if it causes issues, but pandas>=2.0 usually handles it.
        # To be safe and save space, we can drop it since we have 'text_community'.
        drop_cols = [Config.SUBREDDIT_COL]

        train_df.drop(columns=drop_cols, errors="ignore").to_parquet(
            train_cache, index=False
        )
        val_df.drop(columns=drop_cols, errors="ignore").to_parquet(
            val_cache, index=False
        )
        test_df.drop(columns=drop_cols, errors="ignore").to_parquet(
            test_cache, index=False
        )

    return train_df, val_df, test_df
