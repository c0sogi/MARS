import os
import re
import pickle
import random
import numpy as np
import pandas as pd
import scipy.sparse
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def clean_text(text):
    """
    Cleans the input text by:
    1. Converting to lowercase.
    2. Stripping HTML tags.
    3. Removing non-alphanumeric characters (preserving spaces).
    4. Collapsing multiple spaces.

    Args:
        text (str): The raw input text.

    Returns:
        str: The cleaned text.
    """
    if not isinstance(text, str):
        return ""

    # 1. Convert to lowercase
    text = text.lower()

    # 2. Strip HTML tags
    # Replace with space to avoid concatenating words (e.g. "end</p><p>start")
    text = re.sub(r"<[^>]+>", " ", text)

    # 3. Remove non-alphanumeric characters
    # Keep spaces to preserve word tokens.
    # Matches any character that is NOT a-z, 0-9, or whitespace.
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # 4. Collapse multiple spaces and strip
    text = re.sub(r"\s+", " ", text).strip()

    return text


def save_artifact(obj, filepath):
    """
    Saves an artifact to the specified filepath using the appropriate method.
    - scipy.sparse matrices -> .npz
    - pandas DataFrames -> .parquet
    - Other objects -> .pkl (pickle)

    Automatically creates the parent directory if it does not exist.

    Args:
        obj: The object to save.
        filepath (str): The destination path.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if scipy.sparse.issparse(obj):
        # Use scipy's optimized sparse matrix saving
        if not filepath.endswith(".npz"):
            filepath += ".npz"
        scipy.sparse.save_npz(filepath, obj)

    elif isinstance(obj, pd.DataFrame):
        # Use parquet for DataFrames as per data processing requirements
        if not filepath.endswith(".parquet"):
            filepath += ".parquet"
        obj.to_parquet(filepath, index=False)

    else:
        # Default to pickle for models and generic objects
        with open(filepath, "wb") as f:
            pickle.dump(obj, f)


def load_artifact(filepath):
    """
    Loads an artifact from the specified filepath.
    Automatically detects the format based on extension (.npz, .parquet, or pickle).

    Args:
        filepath (str): The path to the artifact.

    Returns:
        The loaded object.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    # Helper to handle cases where extension might be omitted in the call
    if not os.path.exists(filepath):
        if os.path.exists(filepath + ".npz"):
            filepath += ".npz"
        elif os.path.exists(filepath + ".parquet"):
            filepath += ".parquet"
        else:
            raise FileNotFoundError(f"Artifact not found at {filepath}")

    if filepath.endswith(".npz"):
        return scipy.sparse.load_npz(filepath)
    elif filepath.endswith(".parquet"):
        return pd.read_parquet(filepath)
    else:
        with open(filepath, "rb") as f:
            return pickle.load(f)
