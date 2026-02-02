import os
import sys
import random
import logging
import json
import numpy as np
import pandas as pd
import joblib
import torch
from library.config import Config


def setup_logger(name="mf_adbe", log_file=None, level=logging.INFO):
    """
    Sets up a centralized logger that outputs to console and optionally a file.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file.
        level (int): Logging level (e.g., logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # Torch might not be installed or CUDA might not be available
        pass


def save_array(array, path):
    """
    Saves a numpy array to a file, creating directories if necessary.

    Args:
        array (np.ndarray): The array to save.
        path (str): The destination path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, array)


def load_array(path):
    """
    Loads a numpy array from a file.

    Args:
        path (str): The path to load from.

    Returns:
        np.ndarray: The loaded array.
    """
    return np.load(path)


def save_model(model, path):
    """
    Saves a model (or any object) using joblib, creating directories if necessary.

    Args:
        model (object): The object to save.
        path (str): The destination path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    """
    Loads a model using joblib.

    Args:
        path (str): The path to load from.

    Returns:
        object: The loaded model/object.
    """
    return joblib.load(path)


def save_parquet(df, path):
    """
    Saves a DataFrame to a parquet file, creating directories if necessary.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        path (str): The destination path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path):
    """
    Loads a DataFrame from a parquet file.

    Args:
        path (str): The path to load from.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    return pd.read_parquet(path)


def load_data_splits():
    """
    Loads the raw JSON data and merges it with the metadata CSVs to reconstruct
    the Train, Validation, and Test splits as defined in the metadata generation step.

    Returns:
        tuple: (df_train, df_val, df_test)
    """
    # Load Metadata
    meta_train = pd.read_csv(Config.METADATA_TRAIN)
    meta_val = pd.read_csv(Config.METADATA_VAL)
    meta_test = pd.read_csv(Config.METADATA_TEST)

    # Load Raw JSON Data
    with open(Config.INPUT_TRAIN_JSON, "r") as f:
        raw_train_list = json.load(f)

    with open(Config.INPUT_TEST_JSON, "r") as f:
        raw_test_list = json.load(f)

    # Helper to merge metadata with raw data using sample_index
    def merge_data(meta_df, raw_list, source_filename):
        # Filter metadata for the specific source file (though usually consistent)
        subset_meta = meta_df[
            meta_df["source_file"].str.contains(source_filename)
        ].copy()

        # Extract records from raw list based on sample_index
        indices = subset_meta["sample_index"].values
        records = [raw_list[i] for i in indices]

        # Create DataFrame from records
        df_records = pd.DataFrame(records)

        # Ensure alignment (optional but good for safety)
        # We drop columns from raw that might conflict or are redundant if we want to trust metadata
        # But here we mostly want the features from raw.
        # We merge on request_id to be absolutely sure, or just concat if indices are aligned.
        # Merging on request_id is safer.

        # However, df_records might have duplicates if the raw file had them (unlikely here).
        # Let's just merge the metadata columns (like label) onto the raw features.

        merged = pd.merge(
            subset_meta, df_records, on="request_id", how="left", suffixes=("_meta", "")
        )

        # Clean up: If 'requester_received_pizza' is in both, keep the one from metadata (ground truth)
        if "requester_received_pizza_meta" in merged.columns:
            merged["requester_received_pizza"] = merged["requester_received_pizza_meta"]
            merged.drop(columns=["requester_received_pizza_meta"], inplace=True)

        return merged

    # Construct Splits
    # Train and Val come from train.json
    df_train = merge_data(meta_train, raw_train_list, "train.json")
    df_val = merge_data(meta_val, raw_train_list, "train.json")

    # Test comes from test.json
    df_test = merge_data(meta_test, raw_test_list, "test.json")

    return df_train, df_val, df_test
