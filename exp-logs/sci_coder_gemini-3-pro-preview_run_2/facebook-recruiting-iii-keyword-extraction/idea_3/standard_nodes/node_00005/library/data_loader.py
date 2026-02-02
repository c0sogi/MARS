import os
import pandas as pd
from library.config import Config
from library.utils import clean_text


def load_raw_data(split: str, limit: int = None) -> pd.DataFrame:
    """
    Loads the raw CSV data for the specified split from the metadata directory.

    Args:
        split (str): One of 'train', 'val', or 'test'.
        limit (int, optional): Number of rows to load (for debugging).

    Returns:
        pd.DataFrame: The raw dataframe.
    """
    if split == "train":
        path = Config.TRAIN_PATH
    elif split == "val":
        path = Config.VAL_PATH
    elif split == "test":
        path = Config.TEST_PATH
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    # Use 'c' engine for faster CSV reading
    df = pd.read_csv(path, nrows=limit, engine="c")
    return df


def prepare_text_and_tags(df: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
    """
    Preprocesses the dataframe by:
    1. Concatenating 'Title' and 'Body' into a new 'text' column.
    2. Cleaning the text using the utility function.
    3. parsing the 'Tags' string into a list of tags (for train/val).

    Args:
        df (pd.DataFrame): The raw dataframe.
        is_test (bool): Whether the dataframe is the test set (no Tags column).

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    # Ensure Title and Body are strings to handle potential NaNs gracefully
    title = df["Title"].fillna("").astype(str)
    body = df["Body"].fillna("").astype(str)

    # Concatenate Title and Body with a space separator
    raw_text = title + " " + body

    # Clean the concatenated text
    # Note: clean_text handles HTML stripping, lowercasing, and special char removal
    print("Cleaning text data (this may take a while)...")
    df["text"] = raw_text.apply(clean_text)

    # Process Tags if present
    if not is_test:
        # Tags are space-delimited strings in the raw file
        # Convert to list of strings
        df["tags_list"] = df["Tags"].fillna("").astype(str).str.split()

    return df


def load_dataset(
    split: str = "train", limit: int = None, load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Main function to load data with caching support.

    Logic:
    1. If load_cached_data is True, try to load from ./working/idea_3/{split}_processed.parquet.
    2. If not found or load_cached_data is False, load raw data, process it, and save to cache.
    3. If 'limit' is provided, it returns a subset. Note: Partial datasets are NOT saved to the main cache
       to prevent corrupting the full dataset cache.

    Args:
        split (str): 'train', 'val', or 'test'.
        limit (int, optional): Max rows to load.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    # Ensure the working directory exists
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    cache_filename = f"{split}_processed.parquet"
    cache_path = os.path.join(Config.WORK_DIR, cache_filename)

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split} data from cache: {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            if limit is not None:
                df = df.head(limit)
            return df
        except Exception as e:
            print(f"Failed to load cache ({e}). Recomputing from scratch...")

    # 2. Compute from scratch
    print(f"Processing {split} data from scratch...")

    # Load raw data
    # If limit is set, we only load that amount to save time,
    # but we won't save this partial result to the main cache file.
    df = load_raw_data(split, limit)

    # Process text and tags
    is_test = split == "test"
    df = prepare_text_and_tags(df, is_test=is_test)

    # 3. Save to cache
    # Only save if we processed the full dataset (limit is None) to ensure cache integrity
    if limit is None:
        print(f"Saving processed {split} data to cache: {cache_path}")
        df.to_parquet(cache_path, index=False)
    else:
        print(
            f"Limit set to {limit}. Skipping cache save to avoid overwriting full dataset with partial data."
        )

    return df
