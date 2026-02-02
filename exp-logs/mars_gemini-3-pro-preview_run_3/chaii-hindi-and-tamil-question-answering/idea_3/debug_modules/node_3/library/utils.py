import os
import sys
import warnings

# Ensure the library module can be found
sys.path.append(os.getcwd())

from library.config import Config


def set_seed(seed=None):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    Delegates to the centralized Config.set_seed method.

    Args:
        seed (int, optional): The seed to set. If None, uses Config.SEED.
    """
    if seed is None:
        seed = Config.SEED
    Config.set_seed(seed)


def jaccard(str1, str2):
    """
    Computes the word-level Jaccard score between two strings.

    Args:
        str1 (str): The ground truth string.
        str2 (str): The predicted string.

    Returns:
        float: The Jaccard score.
    """
    if str1 is None:
        str1 = ""
    if str2 is None:
        str2 = ""

    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)

    denominator = len(a) + len(b) - len(c)

    if denominator == 0:
        return 0.0

    return float(len(c)) / denominator


def clean_text(text):
    """
    Basic text cleaning utility to ensure inputs are strings and stripped of excess whitespace.

    Args:
        text (str or any): Input text.

    Returns:
        str: Cleaned text.
    """
    if not isinstance(text, str):
        return str(text)
    return text.strip()
