import os
import json
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import setup_logger

# Initialize logger
logger = setup_logger("data_loader")


def _load_raw_json(path):
    """
    Helper to load raw JSON data from disk.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data file not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)
    return data


def _extract_features(entry, is_test=False):
    """
    Extracts relevant features from a single raw data entry.
    """
    # Text Extraction: Concatenate title and text
    # We use space separator. Handling None/NaN by converting to empty string.
    title = str(entry.get("request_title", "") or "")
    text = str(entry.get("request_text_edit_aware", "") or "")
    text_combined = (title + " " + text).strip()

    # Metadata Extraction
    features = {"request_id": entry.get("request_id"), "text_combined": text_combined}

    # Extract numerical metadata defined in Config
    for col in Config.METADATA_COLS:
        val = entry.get(col)
        # Ensure numerical consistency, handle None as NaN
        if val is None:
            features[col] = np.nan
        else:
            features[col] = float(val)

    # Extract Target if not test set
    if not is_test:
        # The target in raw json is boolean, convert to int
        raw_label = entry.get("requester_received_pizza", False)
        features["label"] = 1 if raw_label else 0

    return features


def _process_split(metadata_path, raw_data_path, is_test=False, debug=False):
    """
    Processes a specific data split (train/val/test) by merging metadata with raw JSON.
    """
    logger.info(f"Processing split from metadata: {metadata_path}")

    # Load Metadata
    df_meta = pd.read_csv(metadata_path)

    # Debugging: Subsample if requested
    if debug:
        logger.info(f"Debug mode enabled. Subsampling {Config.DEBUG_SAMPLES} rows.")
        df_meta = df_meta.head(Config.DEBUG_SAMPLES)

    # Load Raw Data
    # We load the full raw file once. Since we have 220GB RAM, this is safe.
    # The metadata 'sample_index' maps directly to the list index in raw json.
    raw_data_list = _load_raw_json(raw_data_path)

    processed_records = []

    # Iterate through metadata and fetch corresponding raw data
    # We use the sample_index from metadata to access the raw list O(1)
    for _, row in df_meta.iterrows():
        idx = int(row["sample_index"])
        if idx >= len(raw_data_list):
            logger.warning(
                f"Sample index {idx} out of bounds for file {raw_data_path}. Skipping."
            )
            continue

        raw_entry = raw_data_list[idx]

        # Verify alignment (optional but good practice)
        if raw_entry.get("request_id") != row["request_id"]:
            logger.error(
                f"ID Mismatch at index {idx}: Meta({row['request_id']}) vs Raw({raw_entry.get('request_id')})"
            )
            raise ValueError("Metadata and Raw Data are misaligned.")

        # Extract features
        record = _extract_features(raw_entry, is_test=is_test)
        processed_records.append(record)

    # Create DataFrame
    df_processed = pd.DataFrame(processed_records)

    # Ensure label is integer if present
    if "label" in df_processed.columns:
        df_processed["label"] = df_processed["label"].astype(int)

    return df_processed


def load_and_process_data(split="train", load_cached_data=True, debug=False):
    """
    Main function to load and process data for a specific split.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): If True, attempts to load from Parquet cache.
        debug (bool): If True, processes a small subset of data.

    Returns:
        pd.DataFrame: Processed dataframe containing features and labels (if applicable).
    """
    # Determine paths based on split
    if split == "train":
        meta_path = Config.TRAIN_META
        raw_path = Config.TRAIN_JSON
        cache_path = Config.TRAIN_FEATURES
        is_test = False
    elif split == "val":
        meta_path = Config.VAL_META
        # Validation comes from train.json in this setup (stratified split)
        raw_path = Config.TRAIN_JSON
        cache_path = Config.VAL_FEATURES
        is_test = False
    elif split == "test":
        meta_path = Config.TEST_META
        raw_path = Config.TEST_JSON
        cache_path = Config.TEST_FEATURES
        is_test = True
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    # Handle Debug Cache Path (avoid overwriting full cache with debug data)
    if debug:
        base, ext = os.path.splitext(cache_path)
        cache_path = f"{base}_debug{ext}"

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading cached {split} data from {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            logger.info(f"Successfully loaded {len(df)} rows from cache.")
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    logger.info(f"Computing {split} data from scratch...")
    df = _process_split(meta_path, raw_path, is_test=is_test, debug=debug)

    # 3. Save to cache
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_parquet(cache_path, index=False)
        logger.info(f"Saved processed {split} data to {cache_path}")
    except Exception as e:
        logger.error(f"Failed to save cache to {cache_path}: {e}")

    return df
