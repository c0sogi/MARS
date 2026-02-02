import os
import random
import re
import unicodedata
import numpy as np
import torch
import pandas as pd
from typing import List, Union, Optional
from library.config import Config


def set_seed(seed: int = Config.SEED) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_semiotic(text: str) -> bool:
    """
    Checks if a token contains semiotic content (digits or Latin characters)
    that typically requires normalization beyond simple lookup.

    Args:
        text: The input string token.

    Returns:
        True if the text matches the semiotic regex, False otherwise.
    """
    if not isinstance(text, str):
        return False
    # Uses the regex from Config: r"[0-9a-zA-Z]"
    return bool(re.search(Config.SEMIOTIC_REGEX, text))


def normalize_string(text: str) -> str:
    """
    Performs standard string normalization.

    1. Converts to Unicode NFC form.
    2. Strips leading/trailing whitespace.

    Args:
        text: Input string.

    Returns:
        Normalized string.
    """
    if not isinstance(text, str):
        return str(text)

    text = unicodedata.normalize("NFC", text)
    return text.strip()


def calculate_accuracy(predictions: List[str], targets: List[str]) -> float:
    """
    Calculates the exact match accuracy between predictions and targets.

    Args:
        predictions: List of predicted strings.
        targets: List of ground truth strings.

    Returns:
        The percentage of correct predictions (0.0 to 1.0).
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"Length mismatch: predictions ({len(predictions)}) vs targets ({len(targets)})"
        )

    if not targets:
        return 0.0

    correct = sum(1 for p, t in zip(predictions, targets) if p == t)
    return correct / len(targets)


def save_parquet_cache(df: pd.DataFrame, path: str) -> None:
    """
    Saves a pandas DataFrame to a parquet file, ensuring the parent directory exists.

    Args:
        df: The DataFrame to save.
        path: The destination file path.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet_cache(path: str) -> pd.DataFrame:
    """
    Loads a pandas DataFrame from a parquet file.

    Args:
        path: The source file path.

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache file not found: {path}")
    return pd.read_parquet(path)
