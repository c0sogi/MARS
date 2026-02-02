import os
import sys
import random
import json
import logging
import numpy as np
import pandas as pd
import torch
import library.config as config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch seeding
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(name, log_file, level=logging.INFO):
    """
    Configures a logger to output to both the console and a file.

    Args:
        name (str): The name of the logger.
        log_file (str): Path to the log file.
        level (int): Logging level (e.g., logging.INFO).

    Returns:
        logging.Logger: The configured logger instance.
    """
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler
    # Ensure directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def load_dataset(split, debug_size=None):
    """
    Loads the dataset for a specific split by combining metadata with raw JSON data.

    Args:
        split (str): One of 'train', 'val', or 'test'.
        debug_size (int, optional): If provided, limits the data to this number of samples
                                    for debugging purposes.

    Returns:
        pd.DataFrame: The loaded dataset containing features and labels (if available).
    """
    # Determine file paths based on split
    if split == "train":
        meta_path = config.TRAIN_META_PATH
        json_path = config.TRAIN_JSON_PATH
    elif split == "val":
        meta_path = config.VAL_META_PATH
        json_path = config.TRAIN_JSON_PATH  # Validation set comes from train.json
    elif split == "test":
        meta_path = config.TEST_META_PATH
        json_path = config.TEST_JSON_PATH
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    # Load Metadata
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df_meta = pd.read_csv(meta_path)

    # Apply Debug Sampling if requested
    if debug_size is not None:
        if debug_size < len(df_meta):
            df_meta = df_meta.iloc[:debug_size].copy()
            # print(f"DEBUG MODE: Sampled {len(df_meta)} rows from {split} set.")

    # Load Raw JSON Data
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Raw data file not found: {json_path}")

    with open(json_path, "r") as f:
        raw_data_list = json.load(f)

    # Efficiently extract only the required rows using sample_index
    # The metadata contains 'sample_index' which corresponds to the list index in the JSON
    indices = df_meta["sample_index"].values
    selected_raw_data = [raw_data_list[i] for i in indices]

    # Create DataFrame from selected raw data
    df_raw = pd.DataFrame(selected_raw_data)

    # Merge metadata with raw data
    # We use the metadata as the base to preserve the split and labels
    # 'request_id' is the common key, but we can also just concatenate since we aligned by index
    # However, merge is safer to ensure ID alignment.

    # Drop potential duplicate columns from raw data that are already in metadata (like label)
    # Metadata is the source of truth for labels in train/val
    cols_to_drop = []
    if (
        "requester_received_pizza" in df_raw.columns
        and "requester_received_pizza" in df_meta.columns
    ):
        cols_to_drop.append("requester_received_pizza")

    if cols_to_drop:
        df_raw = df_raw.drop(columns=cols_to_drop)

    df_final = pd.merge(df_meta, df_raw, on="request_id", how="left")

    return df_final
