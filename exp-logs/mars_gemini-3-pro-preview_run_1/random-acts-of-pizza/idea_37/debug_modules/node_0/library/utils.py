import os
import random
import re
import numpy as np
import pandas as pd
import torch
from library.config import RANDOM_STATE


def set_seed(seed=RANDOM_STATE):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to RANDOM_STATE from config.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set environment variable for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    """
    Returns the PyTorch device (CPU or CUDA) based on availability.

    Returns:
        torch.device: The device object.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def clean_text(text):
    """
    Performs basic string sanitization.

    Steps:
    1. Handles None or NaN values by returning an empty string.
    2. Converts text to lowercase.
    3. Removes URLs.
    4. Replaces newlines/tabs with spaces and strips excessive whitespace.

    Args:
        text (str or object): The input text to clean.

    Returns:
        str: The cleaned text string.
    """
    # Handle None
    if text is None:
        return ""

    # Handle NaN (pandas/numpy)
    try:
        if pd.isna(text):
            return ""
    except:
        pass

    # Ensure string format
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # Replace whitespace (newlines, tabs) with single space
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text
