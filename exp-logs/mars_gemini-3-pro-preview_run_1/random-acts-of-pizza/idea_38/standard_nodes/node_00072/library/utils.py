import os
import json
import random
import numpy as np
import pandas as pd
import torch
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def load_json_data(config, load_cached_data=True):
    """
    Loads the raw JSON datasets and splits them according to the metadata CSVs.
    This ensures that complex data types (like lists of subreddits) are preserved
    from the raw JSON, while adhering to the stratified splits defined in metadata.

    Implements caching using Parquet files to speed up subsequent runs.

    Args:
        config: Configuration class or object containing paths.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Ensure working directory exists
    cache_dir = config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache_path = os.path.join(cache_dir, "train_split.parquet")
    val_cache_path = os.path.join(cache_dir, "val_split.parquet")
    test_cache_path = os.path.join(cache_dir, "test_split.parquet")

    # 1. Try loading from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache_path)
            and os.path.exists(val_cache_path)
            and os.path.exists(test_cache_path)
        ):

            try:
                train_df = pd.read_parquet(train_cache_path)
                val_df = pd.read_parquet(val_cache_path)
                test_df = pd.read_parquet(test_cache_path)
                return train_df, val_df, test_df
            except Exception:
                # If cache loading fails, proceed to re-process
                pass

    # 2. Load Metadata to get the correct splits and source file mappings
    # The metadata CSVs define which request_id belongs to which split
    meta_train = pd.read_csv(config.TRAIN_PATH)
    meta_val = pd.read_csv(config.VAL_PATH)
    meta_test = pd.read_csv(config.TEST_PATH)

    # 3. Helper to load raw data based on metadata
    def _reconstruct_dataframe(meta_df):
        # Identify all unique source files referenced in this split
        source_files = meta_df["source_file"].unique()
        input_root = "./input"

        # Load all referenced JSON files into a lookup dictionary
        # Key: request_id, Value: Full JSON object (dict)
        id_to_record_map = {}

        for rel_path in source_files:
            full_path = os.path.join(input_root, rel_path)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Source file {full_path} not found.")

            with open(full_path, "r") as f:
                data = json.load(f)
                # data is a list of dicts
                for entry in data:
                    if "request_id" in entry:
                        id_to_record_map[entry["request_id"]] = entry

        # Reconstruct the list of records in the exact order of the metadata
        # This ensures alignment with any other metadata-derived features
        ordered_records = []
        missing_ids = []

        for req_id in meta_df["request_id"]:
            if req_id in id_to_record_map:
                ordered_records.append(id_to_record_map[req_id])
            else:
                missing_ids.append(req_id)

        if missing_ids:
            raise ValueError(
                f"Found {len(missing_ids)} IDs in metadata missing from raw JSONs. First few: {missing_ids[:5]}"
            )

        return pd.DataFrame(ordered_records)

    # 4. Construct DataFrames
    train_df = _reconstruct_dataframe(meta_train)
    val_df = _reconstruct_dataframe(meta_val)
    test_df = _reconstruct_dataframe(meta_test)

    # 5. Handle Debug Mode (Subsampling)
    if config.DEBUG:
        train_df = train_df.head(config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(config.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(config.DEBUG_SAMPLE_SIZE)

    # 6. Save to Cache
    # Parquet is used to preserve data types (like lists) better than CSV
    try:
        train_df.to_parquet(train_cache_path)
        val_df.to_parquet(val_cache_path)
        test_df.to_parquet(test_cache_path)
    except Exception as e:
        # If saving fails (e.g. object types not supported), we continue without caching
        # but printing might be restricted, so we just pass
        pass

    return train_df, val_df, test_df


def get_common_columns(train_df, test_df):
    """
    Identifies the intersection of columns between training and test datasets.
    This is crucial for preventing feature leakage where the train set might
    have columns (like 'requester_received_pizza') that are absent in test.

    Args:
        train_df (pd.DataFrame): Training data.
        test_df (pd.DataFrame): Test data.

    Returns:
        list: List of column names present in both DataFrames.
    """
    return list(set(train_df.columns) & set(test_df.columns))


def save_submission(predictions, test_ids, config):
    """
    Formats and saves the predictions to a CSV file for submission.

    Args:
        predictions (array-like): Predicted probabilities of success.
        test_ids (array-like): Corresponding request_ids.
        config: Configuration object containing the output path.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)

    # Create DataFrame
    submission = pd.DataFrame(
        {"request_id": test_ids, "requester_received_pizza": predictions}
    )

    # Save to CSV
    submission.to_csv(config.SUBMISSION_PATH, index=False)
