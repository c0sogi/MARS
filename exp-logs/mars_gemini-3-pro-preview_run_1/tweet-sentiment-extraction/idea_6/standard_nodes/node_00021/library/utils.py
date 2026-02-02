import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.seed):
    """
    Sets the seed for generating random numbers to ensure reproducibility
    across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # deterministic algorithms for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def jaccard(str1, str2):
    """
    Calculates the word-level Jaccard score (intersection over union) between two strings.

    Args:
        str1 (str): The first string (e.g., ground truth).
        str2 (str): The second string (e.g., prediction).

    Returns:
        float: The Jaccard similarity score between 0.0 and 1.0.
    """
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())
    c = a.intersection(b)

    union_len = len(a) + len(b) - len(c)
    return float(len(c)) / union_len if union_len > 0 else 0.0


def process_text(text):
    """
    Normalizes the input text by collapsing multiple whitespaces into a single space
    and stripping leading/trailing whitespace.

    This implements the 'Normalize-First' strategy. It is crucial to run this
    on the raw text before tokenization to ensure that tokenizer offsets map
    correctly to the text used for final extraction.

    Args:
        text (str): The raw input text.

    Returns:
        str: The normalized text.
    """
    if text is None:
        return ""

    # Convert to string to handle potential non-string inputs (e.g. NaN floats)
    s = str(text)

    # split() removes all whitespace characters (space, tab, newline, return, formfeed)
    # and " ".join() puts them back with exactly one space between words.
    return " ".join(s.split())
