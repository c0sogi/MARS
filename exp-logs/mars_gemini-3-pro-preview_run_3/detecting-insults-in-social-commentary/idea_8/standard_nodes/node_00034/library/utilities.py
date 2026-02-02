import os
import random
import codecs
import numpy as np
import torch
import pandas as pd
from library.configuration import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python's random, numpy, and torch.
    Ensures deterministic behavior for CUDA operations.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def decode_text(text):
    """
    Decodes unicode-escaped text strings found in the dataset.

    The dataset contains text where characters are escaped (e.g., "\\n" for newline,
    "\\xe2" for bytes). This function decodes them into proper unicode strings
    to ensure the tokenizer processes the text correctly.

    Args:
        text (str or object): The text to decode.

    Returns:
        str: The decoded string, or an empty string if input is NaN.
    """
    if pd.isna(text):
        return ""

    try:
        # Convert to string and decode escape sequences
        # This handles cases like "Hello\\nWorld" -> "Hello\nWorld"
        return codecs.decode(str(text), "unicode_escape")
    except Exception:
        # Fallback to original string representation if decoding fails
        return str(text)
