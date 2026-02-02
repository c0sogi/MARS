import os
import json
import pandas as pd
from library.config import Config
from library.utils import setup_logger

# Initialize logger
logger = setup_logger("data_loader")


def load_data(load_cached_data: bool = True):
    """
    Loads the training, validation, and test datasets.

    This function handles the merging of raw JSON data with the provided metadata CSVs
    to create aligned DataFrames. It implements caching using Parquet files to speed up
    subsequent executions.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed Parquet files
                                 from the working directory. If False or if files are
                                 missing, re-processes the raw JSON/CSV data.

    Returns:
        tuple: A tuple containing three pandas DataFrames: (train_df, val_df, test_df).
    """
    # Define cache paths
    cache_train_path = os.path.join(Config.WORKING_DIR, "train_merged.parquet")
    cache_val_path = os.path.join(Config.WORKING_DIR, "val_merged.parquet")
    cache_test_path = os.path.join(Config.WORKING_DIR, "test_merged.parquet")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(cache_train_path)
            and os.path.exists(cache_val_path)
            and os.path.exists(cache_test_path)
        ):
            logger.info("Loading datasets from cache...")
            try:
                train_df = pd.read_parquet(cache_train_path)
                val_df = pd.read_parquet(cache_val_path)
                test_df = pd.read_parquet(cache_test_path)
                logger.info("Successfully loaded datasets from cache.")
                return train_df, val_df, test_df
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Re-processing data.")
        else:
            logger.info("Cache files not found. Processing raw data...")
    else:
        logger.info("Ignoring cache. Processing raw data...")

    # 2. Load Raw JSON Data
    logger.info(f"Loading raw train data from {Config.TRAIN_JSON_PATH}...")
    with open(Config.TRAIN_JSON_PATH, "r") as f:
        raw_train_data = json.load(f)

    logger.info(f"Loading raw test data from {Config.TEST_JSON_PATH}...")
    with open(Config.TEST_JSON_PATH, "r") as f:
        raw_test_data = json.load(f)

    # 3. Load Metadata
    logger.info("Loading metadata CSVs...")
    meta_train = pd.read_csv(Config.TRAIN_META_PATH)
    meta_val = pd.read_csv(Config.VAL_META_PATH)
    meta_test = pd.read_csv(Config.TEST_META_PATH)

    # 4. Helper function to merge metadata with raw data
    def merge_data(meta_df, raw_data_source, split_name):
        """
        Merges metadata DataFrame with raw JSON data using sample_index.
        """
        logger.info(f"Constructing {split_name} DataFrame...")

        # Extract the list of indices and verify they are within bounds
        indices = meta_df["sample_index"].values

        # Retrieve records from the raw list using indices
        # This assumes raw_data_source is a list of dicts
        selected_records = [raw_data_source[i] for i in indices]

        # Create DataFrame from records
        data_df = pd.DataFrame(selected_records)

        # Ensure the metadata columns (like label) are preserved/aligned
        # We drop columns from data_df that might conflict or be redundant if needed,
        # but here we primarily want to ensure the target label is correct from metadata
        # for train/val, or just keep the data_df structure.

        # For train/val, metadata has 'requester_received_pizza'.
        # The raw json also has it, but we trust metadata for the split.
        # We will append the metadata info to ensure alignment, though raw JSON usually contains all fields.

        # Verify alignment (optional but good for safety)
        if "request_id" in data_df.columns:
            # Check a few IDs to ensure alignment
            assert (
                data_df["request_id"].iloc[0] == meta_df["request_id"].iloc[0]
            ), f"Mismatch in {split_name} alignment!"

        # If metadata contains the label and we want to enforce it (e.g. if raw is messy),
        # we can overwrite or ensure it exists.
        if "requester_received_pizza" in meta_df.columns:
            data_df["requester_received_pizza"] = meta_df[
                "requester_received_pizza"
            ].values

        return data_df

    # 5. Construct DataFrames
    # Note: Both train and val splits come from the 'train.json' raw file
    train_df = merge_data(meta_train, raw_train_data, "Train")
    val_df = merge_data(meta_val, raw_train_data, "Validation")

    # Test split comes from 'test.json' raw file
    test_df = merge_data(meta_test, raw_test_data, "Test")

    # 6. Save to Cache
    logger.info("Saving processed datasets to cache...")

    # Sanitize mixed-type columns before saving (Cite debug_lesson_1)
    for df in [train_df, val_df, test_df]:
        if "post_was_edited" in df.columns:
            # Convert mixed boolean/timestamp to binary integer (1=Edited, 0=Not Edited)
            # We treat False/None/0 as 0, and Timestamps/True as 1
            df["post_was_edited"] = (
                df["post_was_edited"]
                .fillna(0)
                .apply(lambda x: 1 if x else 0)
                .astype(int)
            )

    try:
        train_df.to_parquet(cache_train_path, index=False)
        val_df.to_parquet(cache_val_path, index=False)
        test_df.to_parquet(cache_test_path, index=False)
        logger.info(f"Saved cache to {Config.WORKING_DIR}")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")

    logger.info(
        f"Data loading complete. Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}"
    )

    return train_df, val_df, test_df
