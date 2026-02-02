import os
import random
import numpy as np
import torch
import re
import string
from library.config import Config


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

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def jaccard(str1, str2):
    """
    Computes the word-level Jaccard score between two strings.
    Implementation provided in the task description.

    Args:
        str1 (str): The prediction string.
        str2 (str): The ground truth string.

    Returns:
        float: The Jaccard similarity score.
    """
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)

    # Avoid division by zero if both sets are empty
    if len(a) + len(b) - len(c) == 0:
        return 0.0

    return float(len(c)) / (len(a) + len(b) - len(c))


def get_average_jaccard(predictions, ground_truths):
    """
    Computes the average Jaccard score for a list of predictions and ground truths.

    Args:
        predictions (list of str): List of predicted answer strings.
        ground_truths (list of str): List of ground truth answer strings.

    Returns:
        float: The average Jaccard score.
    """
    assert len(predictions) == len(
        ground_truths
    ), "Length of predictions and ground truths must match."

    if len(predictions) == 0:
        return 0.0

    scores = [jaccard(p, g) for p, g in zip(predictions, ground_truths)]
    return sum(scores) / len(scores)


def clean_text(text):
    """
    Cleans the text by removing excessive whitespace.
    Useful for post-processing model outputs before scoring or submission.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text.
    """
    if not isinstance(text, str):
        return str(text)

    # Remove leading/trailing whitespace and collapse internal whitespace
    return " ".join(text.strip().split())


def format_prediction_string(text):
    """
    Formats the prediction string for submission.
    Ensures the text is a valid string and handles edge cases like empty predictions.

    Args:
        text (str): The predicted answer text.

    Returns:
        str: Formatted string ready for the CSV.
    """
    if text is None:
        return ""

    text = clean_text(text)

    # The submission format requires quoted strings if they contain delimiters,
    # but pandas to_csv handles quoting automatically.
    # This function ensures we return a clean string object.
    return text
