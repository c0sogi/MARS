import os
import random
import numpy as np
import torch
import pandas as pd
import re
import ast
from library.config import Config


def set_seed(seed=Config.RANDOM_SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.RANDOM_SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clean_text(text):
    """
    Basic text cleaning utility.
    - Handles NaN values
    - Lowercases text
    - Removes newlines, tabs, and carriage returns
    - Collapses multiple spaces into one

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text.
    """
    if pd.isna(text):
        return ""

    # Convert to string just in case
    text = str(text)

    # Lowercase
    text = text.lower()

    # Replace newlines/tabs with space
    text = re.sub(r"[\n\t\r]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def parse_list_column(column_data):
    """
    Parses a pandas Series containing stringified lists (e.g. "['a', 'b']")
    back into actual Python lists. Useful for recovering list structures
    from CSVs.

    Args:
        column_data (pd.Series): Series containing string representations of lists.

    Returns:
        pd.Series: Series containing actual Python lists.
    """

    def safe_eval(x):
        try:
            if pd.isna(x):
                return []
            # Check if it looks like a list
            x = str(x).strip()
            if not (x.startswith("[") and x.endswith("]")):
                return []
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []

    return column_data.apply(safe_eval)


def load_csv_data(split, debug=Config.DEBUG):
    """
    Loads the metadata CSV for the specified split ('train', 'val', 'test').

    Args:
        split (str): One of 'train', 'val', 'test'.
        debug (bool): If True, returns a subsample of the data defined in Config.DEBUG_SAMPLE_SIZE.

    Returns:
        pd.DataFrame: The loaded data.
    """
    if split == "train":
        path = Config.TRAIN_PATH
    elif split == "val":
        path = Config.VAL_PATH
    elif split == "test":
        path = Config.TEST_PATH
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")

    df = pd.read_csv(path)

    if debug:
        # Deterministic sampling for debug
        sample_size = min(len(df), Config.DEBUG_SAMPLE_SIZE)
        df = df.sample(n=sample_size, random_state=Config.RANDOM_SEED).reset_index(
            drop=True
        )

    return df


def save_submission(ids, probabilities, filename=None):
    """
    Formats and saves the submission file in the required format.

    Args:
        ids (list or np.array): Request IDs.
        probabilities (list or np.array): Predicted probabilities (float).
        filename (str, optional): Path to save the file. Defaults to Config.SUBMISSION_PATH.
    """
    if filename is None:
        filename = Config.SUBMISSION_PATH

    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Create DataFrame
    submission_df = pd.DataFrame(
        {"request_id": ids, "requester_received_pizza": probabilities}
    )

    # Ensure correct types
    submission_df["request_id"] = submission_df["request_id"].astype(str)
    submission_df["requester_received_pizza"] = submission_df[
        "requester_received_pizza"
    ].astype(float)

    # Save
    submission_df.to_csv(filename, index=False)
    print(f"Submission saved to {filename}")
