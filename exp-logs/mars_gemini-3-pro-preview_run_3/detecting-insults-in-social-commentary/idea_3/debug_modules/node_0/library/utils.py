import os
import random
import numpy as np
import torch
import codecs
from sklearn.metrics import roc_auc_score


def seed_everything(seed=42):
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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def decode_text(text):
    """
    Decodes unicode-escaped text.
    The dataset contains text fields that are unicode-escaped (e.g., \\n, \\uXXXX).

    Args:
        text (str): The input text to decode.

    Returns:
        str: The decoded text string.
    """
    if text is None:
        return ""

    # Convert to string if not already
    text_str = str(text)

    try:
        # Decode unicode escape sequences
        return codecs.decode(text_str, "unicode_escape")
    except Exception:
        # Fallback to original text if decoding fails
        return text_str


def get_score(y_true, y_pred):
    """
    Calculates the Area Under the Receiver Operating Curve (AUC).

    Args:
        y_true (array-like): True binary labels.
        y_pred (array-like): Target scores/probabilities.

    Returns:
        float: The AUC score.
    """
    return roc_auc_score(y_true, y_pred)
