import os
import random
import numpy as np
import torch
import pandas as pd
import ast
from sklearn.metrics import roc_auc_score


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_score(y_true, y_pred):
    """
    Calculates the Area Under the Receiver Operating Curve (AUC).

    Args:
        y_true (array-like): Ground truth (correct) labels.
        y_pred (array-like): Predicted probabilities.

    Returns:
        float: The AUC score.
    """
    return roc_auc_score(y_true, y_pred)


def clean_text(text):
    """
    Cleans the text column based on dataset analysis.
    The input text is often unicode-escaped and surrounded by double-quotes.

    Args:
        text (str or object): The raw text string.

    Returns:
        str: The cleaned text.
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # Attempt to use literal_eval to handle python-style string escaping and quotes
    # This handles cases like: "User\nComment" -> User\nComment
    try:
        if text.startswith('"') and text.endswith('"'):
            cleaned = ast.literal_eval(text)
            return cleaned
    except (ValueError, SyntaxError):
        pass

    # Fallback cleanup if literal_eval fails
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]

    # Basic unicode unescape if needed
    try:
        text = text.encode("utf-8").decode("unicode_escape")
    except (UnicodeDecodeError, AttributeError):
        pass

    return text
