import os
import random
import ast
import numpy as np
import torch
from library.config import SEED


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def clean_text(text):
    """
    Cleans the text input by handling unicode escapes, removing surrounding quotes,
    and lowercasing.

    Args:
        text (str or object): The raw text from the dataframe.

    Returns:
        str: The cleaned, lowercased text.
    """
    if text is None:
        return ""

    # Convert to string to handle potential non-string types
    text_str = str(text)

    # Handle NaN or empty strings
    if text_str.lower() == "nan" or not text_str.strip():
        return ""

    # Attempt to parse as a Python string literal to handle unicode escapes and quotes
    # The dataset format is described as unicode-escaped text surrounded by double-quotes.
    try:
        # Check if it looks like a quoted string
        if (text_str.startswith('"') and text_str.endswith('"')) or (
            text_str.startswith("'") and text_str.endswith("'")
        ):
            # ast.literal_eval safely evaluates a string containing a Python literal
            parsed = ast.literal_eval(text_str)
            # Ensure the result is a string
            text_str = str(parsed)
    except (ValueError, SyntaxError):
        # Fallback: Manually strip quotes if literal_eval fails
        if text_str.startswith('"') and text_str.endswith('"'):
            text_str = text_str[1:-1]
        elif text_str.startswith("'") and text_str.endswith("'"):
            text_str = text_str[1:-1]

        # Fallback: Attempt manual unicode unescape
        try:
            text_str = text_str.encode("utf-8").decode("unicode_escape")
        except Exception:
            pass

    # Lowercase the text as per the baseline requirement
    return text_str.lower()
