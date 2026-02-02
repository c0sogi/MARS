import os
import re
import torch
import pandas as pd
from library.config import set_seed


def get_device():
    """
    Returns the appropriate PyTorch device (CUDA if available, else CPU).

    Returns:
        torch.device: The device object.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_semiotic(text):
    """
    Determines if a token is 'semiotic' (contains digits or Latin characters).

    In the Hybrid Cascade architecture, these tokens are routed to the
    Neural Network (Tier 2) or require Confidence Gating, as they typically
    represent numbers, dates, money, or foreign entities that standard
    dictionary lookups cannot handle exhaustively.

    Args:
        text (str): The input token text.

    Returns:
        bool: True if the text contains digits (\d) or Latin letters ([a-zA-Z]),
              False otherwise.
    """
    if not isinstance(text, str):
        return False
    # Regex checks for any digit or any Latin letter.
    # This effectively filters out pure Cyrillic words and punctuation.
    return bool(re.search(r"[\d]|[a-zA-Z]", text))


def load_metadata(split="train"):
    """
    Loads the metadata CSV file for a given split from the ./metadata directory.

    This function ensures that all data is loaded as strings to preserve
    formatting (e.g., keeping leading zeros in '01') and handles missing
    values safely.

    Args:
        split (str): The dataset split to load. One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The loaded dataframe with 'before', 'after', and 'class'
                      columns as strings.
    """
    base_path = "./metadata"
    file_path = os.path.join(base_path, f"{split}.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Metadata file not found: {file_path}")

    # Load data with dtype=str to prevent type inference corruption
    df = pd.read_csv(file_path, dtype=str)

    # Fill NaNs with empty strings for text columns to ensure robust string operations
    text_cols = ["before", "after", "class"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")

    return df


def ensure_dir(file_path):
    """
    Ensures that the directory for a given file path exists.

    Args:
        file_path (str): The full path to the file.
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def save_submission(submission_df, output_path):
    """
    Saves the submission dataframe to a CSV file in the correct format.

    Args:
        submission_df (pd.DataFrame): DataFrame containing 'id' and 'after' columns.
        output_path (str): Path to save the CSV.
    """
    ensure_dir(output_path)

    if "id" not in submission_df.columns or "after" not in submission_df.columns:
        raise ValueError("Submission DataFrame must contain 'id' and 'after' columns.")

    # Save to CSV. Pandas handles quoting of strings containing delimiters automatically.
    submission_df.to_csv(output_path, index=False)
