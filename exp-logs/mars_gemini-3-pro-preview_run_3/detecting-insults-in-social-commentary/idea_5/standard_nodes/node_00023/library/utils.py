import os
import random
import codecs
import numpy as np
import torch
import pandas as pd


def seed_everything(seed=42):
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
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def decode_text(text):
    """
    Decodes unicode-escaped text (e.g., "Hello\\nWorld") commonly found in the dataset.
    Handles NaNs by returning an empty string and catches decoding errors by returning
    the original string.

    Args:
        text (str or object): The input text to decode.

    Returns:
        str: The decoded string.
    """
    if pd.isna(text):
        return ""

    text_str = str(text)

    try:
        # The dataset contains python byte literal style escapes (e.g. \\n, \\xe2)
        # codecs.decode with 'unicode_escape' interprets these correctly.
        return codecs.decode(text_str, "unicode_escape")
    except Exception:
        # If decoding fails, return the original string to avoid data loss
        return text_str
