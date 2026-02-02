import os
import random
import codecs
import numpy as np
import torch
import pandas as pd


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def decode_text(text):
    """
    Decodes unicode-escaped text found in the dataset.
    Handles NaN or None values by returning an empty string.

    Args:
        text (str, float, or None): The input text to decode.

    Returns:
        str: The decoded string.
    """
    # Check for missing values using pandas which handles NaN, None, etc.
    if pd.isna(text):
        return ""

    try:
        # Convert to string to ensure compatibility
        s = str(text)
        # Decode unicode escape sequences (e.g., "Hello\\nWorld" -> "Hello\nWorld")
        return codecs.decode(s, "unicode_escape")
    except Exception:
        # In case of decoding errors, return the original string representation
        return str(text)


def get_device():
    """
    Returns the PyTorch device to use for training/inference.

    Returns:
        torch.device: The CUDA device if available, else CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
