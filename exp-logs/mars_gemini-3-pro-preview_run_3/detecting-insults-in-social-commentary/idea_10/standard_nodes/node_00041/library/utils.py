import os
import random
import numpy as np
import torch
import codecs
import pandas as pd


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def decode_text(text):
    """
    Decodes unicode-escaped text sequences found in the raw dataset.

    The dataset contains strings where characters are represented by their
    unicode escape codes (e.g., '\\n' for newline, '\\xe2' for bytes).
    This function converts them back to standard unicode characters.

    Args:
        text (str or object): The text to decode.

    Returns:
        str: The decoded text string. Returns an empty string if input is NaN.
    """
    if pd.isna(text):
        return ""

    # Convert to string first to handle potential non-string types safely
    text_str = str(text)

    try:
        # The dataset uses python-style unicode escapes (e.g. "Hello\\nWorld")
        # codecs.decode with 'unicode_escape' interprets these correctly.
        decoded = codecs.decode(text_str, "unicode_escape")
        return decoded
    except Exception:
        # Fallback: return original string if decoding fails
        return text_str
