import os
import random
import numpy as np
import torch
import pandas as pd
import re
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def is_semiotic(text):
    """
    Determines if a token requires neural normalization (Tier 2).
    According to the strategy, tokens containing digits or Latin characters
    are considered 'semiotic' or ambiguous and are routed to the Transformer
    if not found in the HFBB memory.

    Args:
        text (str): The input token text.

    Returns:
        bool: True if text contains digits or Latin characters, False otherwise.
    """
    if not isinstance(text, str):
        return False

    # Regex checks for any digit (0-9) or any Latin letter (a-z, A-Z)
    # This covers cases like "$3.16", "Julius", "ISBN-13", etc.
    return bool(re.search(r"[0-9a-zA-Z]", text))


def load_data(split="train"):
    """
    Loads the dataset splits from the metadata directory as defined in Config.

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The loaded dataset with correct data types.
    """
    if split == "train":
        filepath = Config.TRAIN_FILE
    elif split == "val":
        filepath = Config.VAL_FILE
    elif split == "test":
        filepath = Config.TEST_FILE
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Metadata file not found at: {filepath}")

    # Define dtypes to ensure text is read as string.
    # This is crucial for tokens like "007" which pandas might parse as integer 7.
    dtype_map = {"sentence_id": str, "token_id": int, "before": str}

    # Train and Val sets include targets and classes
    if split in ["train", "val"]:
        dtype_map.update({"class": str, "after": str})

    df = pd.read_csv(filepath, dtype=dtype_map)

    # Handle potential NaNs in text columns (e.g., if token is literally "nan" or empty)
    df["before"] = df["before"].fillna("")

    if "after" in df.columns:
        df["after"] = df["after"].fillna("")

    if "class" in df.columns:
        df["class"] = df["class"].fillna("UNKNOWN")

    return df
