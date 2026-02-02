import os
import json
import pandas as pd
import numpy as np
from library import config
from library import utils

# Initialize Logger
logger = utils.setup_logger(
    "data_loader", os.path.join(config.WORKING_DIR, "data_loader.log")
)


def extract_text_fields(df):
    """
    Extracts and cleans text fields from the dataframe.
    Creates a 'text_concat' column for the Global Context view.

    Args:
        df (pd.DataFrame): The input dataframe containing raw text columns.

    Returns:
        pd.DataFrame: A dataframe containing cleaned title, body, and concatenated text.
    """
    # Ensure text columns are strings and handle NaNs
    title = df[config.TEXT_COL_TITLE].fillna("").astype(str)
    body = df[config.TEXT_COL_BODY].fillna("").astype(str)

    # Create concatenated text for the MPNet backbone (Global View)
    text_concat = title + " " + body

    # Return a DataFrame with the processed text fields
    # We preserve the index of the input dataframe
    return pd.DataFrame(
        {
            config.TEXT_COL_TITLE: title,
            config.TEXT_COL_BODY: body,
            "text_concat": text_concat,
        },
        index=df.index,
    )


def extract_metadata(df):
    """
    Extracts and cleans numerical metadata features defined in the configuration.

    Args:
        df (pd.DataFrame): The input dataframe containing raw metadata columns.

    Returns:
        pd.DataFrame: A dataframe containing only the selected numerical features.
    """
    # Identify which requested features exist in the dataframe
    available_features = [f for f in config.NUMERIC_FEATURES if f in df.columns]

    if len(available_features) < len(config.NUMERIC_FEATURES):
        missing = set(config.NUMERIC_FEATURES) - set(available_features)
        logger.warning(
            f"The following numeric features were not found in data: {missing}"
        )

    # Extract features
    features_df = df[available_features].copy()

    # Fill missing values with 0 (safe assumption for counts/scores/timestamps in this domain)
    features_df = features_df.fillna(0)

    return features_df


def _load_raw_json(path):
    """Helper to load raw JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data file not found at {path}")
    with open(path, "r") as f:
        return json.load(f)


def _process_split(meta_path, raw_data, is_test=False, debug=False):
    """
    Merges metadata with raw JSON data based on sample_index.
    """
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found at {meta_path}")

    meta_df = pd.read_csv(meta_path)

    # Apply debug slicing if enabled
    if debug:
        meta_df = meta_df.head(config.DEBUG_SAMPLE_SIZE)

    # Map metadata to raw data using sample_index
    # raw_data is a list of dictionaries
    indices = meta_df["sample_index"].values
    records = [raw_data[i] for i in indices]

    # Create DataFrame from the selected records
    df = pd.DataFrame(records)

    # Ensure the request_id matches (sanity check) and is preserved
    df["request_id"] = meta_df["request_id"].values

    # Merge target label from metadata for training/validation sets
    if not is_test:
        df[config.TARGET_COL] = meta_df[config.TARGET_COL].values

    return df


def load_dataset(load_cached_data=True, debug_mode=config.DEBUG_MODE):
    """
    Main function to load train, validation, and test datasets.
    Handles caching to speed up subsequent runs.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache.
        debug_mode (bool): If True, loads only a small subset of data.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Define cache paths
    cache_train = os.path.join(config.WORKING_DIR, "train_processed.parquet")
    cache_val = os.path.join(config.WORKING_DIR, "val_processed.parquet")
    cache_test = os.path.join(config.WORKING_DIR, "test_processed.parquet")

    # Check if cache exists and we want to load it
    # Note: If in debug mode, we ignore the full cache to ensure we return the small subset
    if load_cached_data and not debug_mode:
        if (
            os.path.exists(cache_train)
            and os.path.exists(cache_val)
            and os.path.exists(cache_test)
        ):
            logger.info("Loading processed datasets from cache...")
            train_df = pd.read_parquet(cache_train)
            val_df = pd.read_parquet(cache_val)
            test_df = pd.read_parquet(cache_test)
            return train_df, val_df, test_df

    logger.info("Cache not found or invalid. Processing raw data...")

    # Load raw data into memory
    logger.info(f"Loading raw train JSON from {config.TRAIN_JSON_PATH}...")
    train_raw = _load_raw_json(config.TRAIN_JSON_PATH)

    logger.info(f"Loading raw test JSON from {config.TEST_JSON_PATH}...")
    test_raw = _load_raw_json(config.TEST_JSON_PATH)

    # Process splits using metadata
    logger.info("Processing train split...")
    train_df = _process_split(
        config.TRAIN_META_PATH, train_raw, is_test=False, debug=debug_mode
    )

    logger.info("Processing validation split...")
    val_df = _process_split(
        config.VAL_META_PATH, train_raw, is_test=False, debug=debug_mode
    )

    logger.info("Processing test split...")
    test_df = _process_split(
        config.TEST_META_PATH, test_raw, is_test=True, debug=debug_mode
    )

    # Apply feature extraction/cleaning to all splits
    for df in [train_df, val_df, test_df]:
        # Extract and clean text fields
        text_df = extract_text_fields(df)
        for col in text_df.columns:
            df[col] = text_df[col]

        # Extract and clean numeric metadata
        meta_df_feats = extract_metadata(df)
        for col in meta_df_feats.columns:
            df[col] = meta_df_feats[col]

        # Fix mixed-type columns for Parquet serialization
        # 'post_was_edited' contains mixed boolean/float types. We cast to string.
        if "post_was_edited" in df.columns:
            df["post_was_edited"] = df["post_was_edited"].astype(str)

    # Save to cache if not in debug mode
    if not debug_mode:
        logger.info("Saving processed datasets to cache...")
        # Ensure working directory exists
        os.makedirs(config.WORKING_DIR, exist_ok=True)

        train_df.to_parquet(cache_train)
        val_df.to_parquet(cache_val)
        test_df.to_parquet(cache_test)

    logger.info(
        f"Data loading complete. Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}"
    )

    return train_df, val_df, test_df
