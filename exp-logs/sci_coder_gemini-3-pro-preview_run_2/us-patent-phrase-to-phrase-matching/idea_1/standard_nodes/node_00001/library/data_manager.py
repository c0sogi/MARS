import os
import pandas as pd
from library.config import Config


def load_csv(file_path: str) -> pd.DataFrame:
    """
    Reads a CSV file into a pandas DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at: {file_path}")
    return pd.read_csv(file_path)


def preprocess_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Formats the text data according to the model requirements.

    Logic:
    1. Concatenates 'anchor' and 'context' with ' [SEP] ' separator.
    2. Renames columns to 'anchor_input' and 'target_input' for clarity.
    3. Retains 'id' and 'score' (if present).
    """
    df_processed = df.copy()

    # Concatenate anchor and context
    # We use " [SEP] " as a separator to indicate the boundary to the tokenizer
    df_processed["anchor_input"] = (
        df_processed["anchor"].astype(str)
        + " [SEP] "
        + df_processed["context"].astype(str)
    )

    # Target remains as is, but we rename it for consistency
    df_processed["target_input"] = df_processed["target"].astype(str)

    # Select relevant columns
    cols_to_keep = ["id", "anchor_input", "target_input"]
    if "score" in df_processed.columns:
        cols_to_keep.append("score")

    return df_processed[cols_to_keep]


def get_data(split: str, load_cached_data: bool = True) -> pd.DataFrame:
    """
    Retrieves the dataset for a specific split ('train', 'val', 'test').
    Implements caching logic to speed up subsequent runs.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        pd.DataFrame: The processed dataset.
    """
    # Determine raw file path
    if split == "train":
        raw_path = Config.TRAIN_PATH
    elif split == "val":
        raw_path = Config.VAL_PATH
    elif split == "test":
        raw_path = Config.TEST_PATH
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    # Determine cache file path
    # We append _debug to the filename if in debug mode to avoid mixing full/partial datasets
    suffix = "_debug" if Config.DEBUG else ""
    cache_filename = f"{split}_processed{suffix}.parquet"
    cache_path = os.path.join(Config.CACHE_DIR, cache_filename)

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        # print(f"Loading cached {split} data from {cache_path}")
        return pd.read_parquet(cache_path)

    # 2. If loading fails or not requested, process from scratch
    # print(f"Processing {split} data from scratch...")

    # Load raw data
    df = load_csv(raw_path)

    # Apply Debugging limit if enabled
    if Config.DEBUG:
        df = df.iloc[: Config.DEBUG_SIZE].copy()

    # Preprocess
    df_processed = preprocess_inputs(df)

    # Save to cache
    df_processed.to_parquet(cache_path, index=False)
    # print(f"Saved processed {split} data to {cache_path}")

    return df_processed
