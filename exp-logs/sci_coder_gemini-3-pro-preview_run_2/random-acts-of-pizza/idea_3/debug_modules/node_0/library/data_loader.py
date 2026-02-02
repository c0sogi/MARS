import os
import json
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import setup_logger, timer


def load_data(load_cached_data=True):
    """
    Loads, merges, and cleans the dataset based on provided metadata and raw JSON files.

    Args:
        load_cached_data (bool): If True, attempts to load processed data from parquet cache.
                                 If False or cache missing, re-processes raw data.

    Returns:
        tuple: (df_train, df_val, df_test) containing the processed pandas DataFrames.
    """
    logger = setup_logger(name="data_loader")

    # Ensure working directory exists for caching
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache paths for cleaned data
    train_cache = os.path.join(Config.WORKING_DIR, "train_cleaned.parquet")
    val_cache = os.path.join(Config.WORKING_DIR, "val_cleaned.parquet")
    test_cache = os.path.join(Config.WORKING_DIR, "test_cleaned.parquet")

    # Attempt to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            with timer("Load Data from Cache", logger):
                try:
                    df_train = pd.read_parquet(train_cache)
                    df_val = pd.read_parquet(val_cache)
                    df_test = pd.read_parquet(test_cache)
                    logger.info(
                        f"Loaded cached data - Train: {df_train.shape}, Val: {df_val.shape}, Test: {df_test.shape}"
                    )
                    return df_train, df_val, df_test
                except Exception as e:
                    logger.warning(
                        f"Failed to load cache: {e}. Proceeding to re-process raw data."
                    )
        else:
            logger.info("Cache files not found. Processing raw data.")

    with timer("Process Raw Data", logger):
        # Load Metadata
        logger.info("Loading metadata...")
        df_meta_train = pd.read_csv(Config.TRAIN_META_PATH)
        df_meta_val = pd.read_csv(Config.VAL_META_PATH)
        df_meta_test = pd.read_csv(Config.TEST_META_PATH)

        # Load Raw JSON Data
        logger.info("Loading raw JSON...")
        with open(Config.RAW_TRAIN_PATH, "r") as f:
            raw_train_data = json.load(f)
        df_raw_train = pd.DataFrame(raw_train_data)

        with open(Config.RAW_TEST_PATH, "r") as f:
            raw_test_data = json.load(f)
        df_raw_test = pd.DataFrame(raw_test_data)

        # Merge Metadata with Raw Data
        # Train and Val are subsets of train.json
        logger.info("Merging metadata with raw data...")
        df_train = df_meta_train.merge(df_raw_train, on="request_id", how="left")
        df_val = df_meta_val.merge(df_raw_train, on="request_id", how="left")

        # Test is from test.json
        df_test = df_meta_test.merge(df_raw_test, on="request_id", how="left")

        # Resolve Target Column Conflicts (if any)
        # Metadata has the authoritative label. Raw data might have it too, leading to _x/_y suffixes.
        for df in [df_train, df_val]:
            if "requester_received_pizza_x" in df.columns:
                df["requester_received_pizza"] = df["requester_received_pizza_x"]
                df.drop(
                    columns=[
                        "requester_received_pizza_x",
                        "requester_received_pizza_y",
                    ],
                    inplace=True,
                    errors="ignore",
                )

            # Ensure target is integer
            if "requester_received_pizza" in df.columns:
                df["requester_received_pizza"] = df["requester_received_pizza"].astype(
                    int
                )

        # Basic Cleaning
        logger.info("Cleaning data...")
        text_cols = ["request_text", "request_text_edit_aware", "request_title"]

        for df in [df_train, df_val, df_test]:
            # 1. Fill NaNs in text columns
            for col in text_cols:
                if col in df.columns:
                    df[col] = df[col].fillna("").astype(str)

            # 2. Handle specific categorical NaNs
            if "requester_user_flair" in df.columns:
                df["requester_user_flair"] = df["requester_user_flair"].fillna("None")

        # Debug Sampling
        if Config.DEBUG:
            logger.info(f"DEBUG Mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
            df_train = df_train.iloc[: Config.DEBUG_SAMPLE_SIZE]
            df_val = df_val.iloc[: Config.DEBUG_SAMPLE_SIZE]
            df_test = df_test.iloc[: Config.DEBUG_SAMPLE_SIZE]

        # Save to Cache
        logger.info(f"Saving processed data to {Config.WORKING_DIR}...")
        df_train.to_parquet(train_cache, index=False)
        df_val.to_parquet(val_cache, index=False)
        df_test.to_parquet(test_cache, index=False)

        logger.info(
            f"Processed data - Train: {df_train.shape}, Val: {df_val.shape}, Test: {df_test.shape}"
        )

    return df_train, df_val, df_test
