import os
import pandas as pd
import numpy as np
from library.utils import clean_html_text

# Define directories
METADATA_DIR = "./metadata"
CACHE_DIR = "./working/idea_2"


def load_metadata(split: str) -> pd.DataFrame:
    """
    Loads the metadata CSV for the specified split.
    Handles mapping 'val' to 'validation.csv'.
    """
    # Map 'val' shorthand to the actual filename 'validation'
    filename = "validation" if split == "val" else split

    file_path = os.path.join(METADATA_DIR, f"{filename}.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Metadata file not found: {file_path}")

    # Load data
    # engine='c' is generally faster for CSVs
    df = pd.read_csv(file_path, engine="c")
    return df


def prepare_input_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the input text by concatenating Title and Body and cleaning them.

    Args:
        df (pd.DataFrame): Dataframe containing 'Title' and 'Body'.

    Returns:
        pd.DataFrame: Dataframe with an added 'text' column.
    """
    # Ensure columns exist
    if "Title" not in df.columns or "Body" not in df.columns:
        raise ValueError("DataFrame must contain 'Title' and 'Body' columns.")

    # Fill NA values to avoid errors during concatenation
    title = df["Title"].fillna("").astype(str)
    body = df["Body"].fillna("").astype(str)

    # Concatenate Title and Body with a space separator
    raw_text = title + " " + body

    # Apply cleaning function from library.utils
    # We use map/apply. Since clean_html_text is deterministic, this is safe.
    print("Cleaning text data (removing HTML, non-alphanumeric)...")
    df["text"] = raw_text.apply(clean_html_text)

    return df


def get_processed_data(
    split: str, limit: int = None, load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Retrieves processed data for a given split, using caching to speed up subsequent calls.

    Args:
        split (str): The dataset split ('train', 'val', 'test').
        limit (int, optional): If set, returns only the first N rows.
        load_cached_data (bool): If True, attempts to load from the cache directory.

    Returns:
        pd.DataFrame: The processed DataFrame containing 'Id', 'text', and 'Tags' (if available).
    """
    # Ensure the cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Define cache filename
    cache_path = os.path.join(CACHE_DIR, f"{split}_processed.parquet")

    df = None

    # 1. Attempt to load from cache
    if load_cached_data:
        if os.path.exists(cache_path):
            print(f"Loading cached data for '{split}' from {cache_path}...")
            try:
                df = pd.read_parquet(cache_path)
            except Exception as e:
                print(f"Failed to load cache: {e}. Proceeding to re-process.")
                df = None
        else:
            print(f"No cache found for '{split}' at {cache_path}.")

    # 2. Process from scratch if needed
    if df is None:
        print(f"Processing data for '{split}' from scratch...")

        # Load raw metadata
        df_raw = load_metadata(split)

        # Process text
        # We process the FULL dataset to ensure the cache file is complete and valid.
        df_processed = prepare_input_text(df_raw)

        # Filter columns to minimize memory usage
        cols_to_keep = ["Id", "text"]
        if "Tags" in df_processed.columns:
            cols_to_keep.append("Tags")

        df = df_processed[cols_to_keep]

        # Save to cache
        print(f"Saving processed data to {cache_path}...")
        df.to_parquet(cache_path, index=False)

    # 3. Apply limit if requested
    if limit is not None:
        print(f"Limiting dataset to {limit} rows.")
        df = df.head(limit)

    return df
