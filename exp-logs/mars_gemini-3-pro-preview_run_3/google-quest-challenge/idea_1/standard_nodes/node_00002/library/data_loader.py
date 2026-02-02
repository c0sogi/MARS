import os
import pandas as pd
import numpy as np
from library.config import CACHE_DIR, TEXT_COLS, TARGET_COLS, ID_COL


def load_dataset(file_path):
    """
    Loads a dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path)


def prepare_text_pairs(df, split_name, load_cached_data=True):
    """
    Preprocesses text columns into question and answer pairs.
    Concatenates question title and body. Handles caching to speed up subsequent runs.

    Args:
        df (pd.DataFrame): Input dataframe containing raw text columns.
        split_name (str): Name of the split (e.g., 'train', 'val', 'test') for naming cache files.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (questions, answers) where both are numpy arrays of strings.
    """
    # Define cache file path
    cache_path = os.path.join(CACHE_DIR, f"{split_name}_processed_text.parquet")

    # 1. IF load_cached_data is True: Try to load the file.
    if load_cached_data and os.path.exists(cache_path):
        try:
            cached_df = pd.read_parquet(cache_path)
            # Verify columns exist
            if "question" in cached_df.columns and "answer" in cached_df.columns:
                print(f"Loaded processed text for '{split_name}' from cache.")
                return cached_df["question"].values, cached_df["answer"].values
        except Exception as e:
            print(f"Failed to load cache for {split_name}: {e}. Reprocessing...")

    # 2. IF loading fails OR load_cached_data is False: Compute/process.
    print(f"Processing text data for '{split_name}'...")

    # Ensure text columns exist in the dataframe, fill with empty if missing
    for col in TEXT_COLS:
        if col not in df.columns:
            df[col] = ""

    # Fill NaNs with empty strings and ensure string type
    df_filled = df[TEXT_COLS].fillna("").astype(str)

    # Process Question: Title + " " + Body
    # Using a space separator
    questions = (
        df_filled["question_title"].str.strip()
        + " "
        + df_filled["question_body"].str.strip()
    ).values

    # Process Answer: Just the answer text
    answers = df_filled["answer"].str.strip().values

    # Create a DataFrame for caching
    processed_df = pd.DataFrame({"question": questions, "answer": answers})

    # Save the result to the cache directory
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        processed_df.to_parquet(cache_path, index=False)
        print(f"Saved processed text for '{split_name}' to cache.")
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")

    # 3. Return the data.
    return questions, answers


def get_targets(df):
    """
    Extracts target values from the dataframe.

    Args:
        df (pd.DataFrame): Dataframe containing target columns.

    Returns:
        np.ndarray: Array of shape (N, 30) containing target values.
    """
    # Check if all target columns exist
    missing_cols = [col for col in TARGET_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing target columns in dataframe: {missing_cols}")

    return df[TARGET_COLS].values


def get_ids(df):
    """
    Extracts the ID column.

    Args:
        df (pd.DataFrame): Dataframe.

    Returns:
        np.ndarray: Array of IDs.
    """
    if ID_COL not in df.columns:
        raise ValueError(f"ID column '{ID_COL}' not found.")
    return df[ID_COL].values
