import os
import json
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def preprocess_text(text, max_length=1000):
    """
    Cleans and truncates source code or markdown text.
    Removes excessive whitespace and limits the character count.

    Args:
        text (str): The input text to process.
        max_length (int): The maximum number of characters to retain.

    Returns:
        str: The cleaned and truncated text.
    """
    if not isinstance(text, str):
        return ""

    # Normalize whitespace: replaces newlines, tabs, and multiple spaces with a single space
    cleaned_text = " ".join(text.split())

    # Truncate to the specified maximum length
    if len(cleaned_text) > max_length:
        cleaned_text = cleaned_text[:max_length]

    return cleaned_text


def read_notebook(path):
    """
    Safely parses a JSON notebook file into a dictionary.

    Args:
        path (str): The file path to the notebook JSON.

    Returns:
        dict: The dictionary containing 'cell_type' and 'source' keys,
              or None if the file cannot be read.
    """
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        # In a real pipeline, we might log this error.
        # For now, we return None to indicate failure safely.
        return None
