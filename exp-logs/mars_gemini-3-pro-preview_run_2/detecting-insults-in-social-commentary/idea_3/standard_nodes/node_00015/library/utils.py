import os
import random
import ast
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score


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


def clean_text(text):
    """
    Cleans the text content by handling unicode escapes and removing surrounding quotes.
    The input text is often a string representation of a string (e.g., '"content"').

    Args:
        text (str): The raw text string.

    Returns:
        str: The cleaned text.
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # Attempt to use literal_eval to handle python-style string escaping and quotes
    # This is effective for inputs like: "some text\nwith newlines"
    try:
        if text.startswith('"') and text.endswith('"'):
            cleaned = ast.literal_eval(text)
            return cleaned
    except (ValueError, SyntaxError):
        pass

    # Fallback cleanup if literal_eval fails
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]

    # Basic unicode unescape if needed (e.g. \xe2 -> char)
    try:
        text = text.encode("utf-8").decode("unicode_escape")
    except (ValueError, UnicodeDecodeError):
        pass

    return text


def calculate_auc(y_true, y_pred):
    """
    Computes the Area Under the Receiver Operating Curve (AUC).

    Args:
        y_true (array-like): Ground truth labels (binary 0/1).
        y_pred (array-like): Predicted probabilities.

    Returns:
        float: The AUC score. Returns 0.5 if only one class is present in y_true.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Check if both classes are present to avoid sklearn errors
    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(y_true, y_pred)
