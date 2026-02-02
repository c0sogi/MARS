import os
import json
import pandas as pd
from library.config import Config
from library.utils import ensure_directory


def load_data(load_from_cache: bool = True):
    """
    Loads the training and testing data.

    This function:
    1. Checks for cached parquet files.
    2. If not found or load_from_cache is False:
       - Loads raw JSON files.
       - Loads metadata CSV files.
       - Combines train and validation metadata to form a full training set.
       - Merges metadata with raw JSON data based on request_id.
       - Caches the result to parquet files.
    3. Returns the training and testing DataFrames.

    Args:
        load_from_cache (bool): Whether to try loading from pre-saved parquet files.

    Returns:
        tuple: (df_train, df_test)
            - df_train: DataFrame containing merged training and validation data.
            - df_test: DataFrame containing test data.
    """

    # Define cache paths
    train_cache_path = os.path.join(Config.WORKING_DIR, "train_merged.parquet")
    test_cache_path = os.path.join(Config.WORKING_DIR, "test_merged.parquet")

    # Attempt to load from cache
    if (
        load_from_cache
        and os.path.exists(train_cache_path)
        and os.path.exists(test_cache_path)
    ):
        # print("Loading data from cache...")
        df_train = pd.read_parquet(train_cache_path)
        df_test = pd.read_parquet(test_cache_path)
    else:
        # print("Loading data from raw files...")

        # 1. Load Metadata
        if (
            not os.path.exists(Config.TRAIN_META_PATH)
            or not os.path.exists(Config.VAL_META_PATH)
            or not os.path.exists(Config.TEST_META_PATH)
        ):
            raise FileNotFoundError(
                "Metadata files not found. Ensure metadata generation script has run."
            )

        df_meta_train = pd.read_csv(Config.TRAIN_META_PATH)
        df_meta_val = pd.read_csv(Config.VAL_META_PATH)
        df_meta_test = pd.read_csv(Config.TEST_META_PATH)

        # Combine train and validation metadata for full CV
        df_meta_full_train = pd.concat([df_meta_train, df_meta_val], ignore_index=True)

        # 2. Load Raw JSON Data
        with open(Config.TRAIN_JSON_PATH, "r") as f:
            raw_train_data = json.load(f)
        df_raw_train = pd.DataFrame(raw_train_data)

        with open(Config.TEST_JSON_PATH, "r") as f:
            raw_test_data = json.load(f)
        df_raw_test = pd.DataFrame(raw_test_data)

        # 3. Merge Metadata with Raw Data
        # We merge on 'request_id'.
        # Note: The raw data contains the target 'requester_received_pizza'.
        # The metadata also contains it. We'll prioritize the one in the dataframe
        # but ensure no duplication of columns.

        # Merge Train (Inner join ensures we only get records listed in metadata)
        df_train = df_meta_full_train.merge(
            df_raw_train, on="request_id", how="inner", suffixes=("", "_raw")
        )

        # Clean up duplicate target column if it exists
        if "requester_received_pizza_raw" in df_train.columns:
            df_train = df_train.drop(columns=["requester_received_pizza_raw"])

        # Merge Test
        df_test = df_meta_test.merge(
            df_raw_test, on="request_id", how="inner", suffixes=("", "_raw")
        )

        # Fix mixed types for Parquet serialization
        # 'post_was_edited' contains False (bool) and timestamps (float).
        for df in [df_train, df_test]:
            if "post_was_edited" in df.columns:
                # Map False -> 0, anything else (timestamp) -> 1
                df["post_was_edited"] = (
                    df["post_was_edited"]
                    .apply(lambda x: 0 if x is False else 1)
                    .astype(int)
                )

        # 4. Cache the results
        ensure_directory(train_cache_path)
        df_train.to_parquet(train_cache_path, index=False)
        df_test.to_parquet(test_cache_path, index=False)

    # Handle Debug Mode
    if Config.DEBUG:
        # print(f"Debug mode enabled. Sampling {Config.MAX_SAMPLES} rows.")
        if len(df_train) > Config.MAX_SAMPLES:
            df_train = df_train.sample(
                n=Config.MAX_SAMPLES, random_state=Config.SEED
            ).reset_index(drop=True)
        if len(df_test) > Config.MAX_SAMPLES:
            df_test = df_test.sample(
                n=Config.MAX_SAMPLES, random_state=Config.SEED
            ).reset_index(drop=True)

    return df_train, df_test
