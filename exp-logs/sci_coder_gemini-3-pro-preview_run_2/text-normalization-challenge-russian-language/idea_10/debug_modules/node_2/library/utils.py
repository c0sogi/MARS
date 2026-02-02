import os
import random
import numpy as np
import torch
import re
import pandas as pd
from library import config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def is_semiotic(text, token_class=None):
    """
    Determines if a token is 'semiotic' (requires normalization via neural network or complex rules).

    Args:
        text (str): The raw token text.
        token_class (str, optional): The class label from the dataset (e.g., 'DATE', 'PLAIN').
                                     If provided, uses the class list from config.

    Returns:
        bool: True if semiotic, False otherwise.
    """
    # Ensure text is a string
    text = str(text)

    # 1. If class is known (Training/Validation), use the explicit list from config
    if token_class is not None and token_class != "UNKNOWN":
        return token_class in config.SEMIOTIC_CLASSES

    # 2. If class is unknown (Inference), use heuristics based on text content.
    # Logic: Contains digits, Latin characters, or special symbols.

    # Check for digits or Latin characters (strong indicators of semiotic content like numbers, units, transliteration)
    if re.search(r"[\d]|[a-zA-Z]", text):
        return True

    # Check for special symbols that are not standard punctuation.
    # We define "Standard" as Cyrillic letters, whitespace, and common punctuation marks.
    # If a character falls outside this set, it is likely a symbol (e.g., $, %, +, =, etc.)
    # Excluded set (Safe):
    #   а-яА-ЯёЁ : Cyrillic alphabet
    #   \s       : Whitespace
    #   \.,:;!\?\(\)\"\'\-\—«» : Standard punctuation and quotes
    if re.search(r"[^а-яА-ЯёЁ\s\.,:;!?\(\)\"\'\-\—«»]", text):
        return True

    return False


def save_cache(df, path):
    """
    Saves a pandas DataFrame to a parquet file.
    Ensures the parent directory exists.

    Args:
        df (pd.DataFrame): The dataframe to save.
        path (str): The destination file path.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    df.to_parquet(path, index=False)


def load_cache(path):
    """
    Loads a pandas DataFrame from a parquet file.

    Args:
        path (str): The file path to load.

    Returns:
        pd.DataFrame or None: The loaded dataframe, or None if the file does not exist.
    """
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None
