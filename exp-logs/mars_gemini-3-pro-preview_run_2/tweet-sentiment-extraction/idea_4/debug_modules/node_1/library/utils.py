import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

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


def jaccard(str1, str2):
    """
    Computes the Jaccard score between two strings.

    The Jaccard score is defined as the size of the intersection divided by the
    size of the union of the sample sets. Here, the sets are created by splitting
    the strings by whitespace and converting to lowercase.

    Args:
        str1 (str): The first string (e.g., predicted text).
        str2 (str): The second string (e.g., ground truth text).

    Returns:
        float: The Jaccard similarity score (0.0 to 1.0).
    """
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())
    c = a.intersection(b)

    union_len = len(a) + len(b) - len(c)
    if union_len == 0:
        return 0.0
    return float(len(c)) / union_len


def get_selected_text(text, start_idx, end_idx, offsets):
    """
    Reconstructs the selected text from the original text using token indices
    and the tokenizer's offset mapping.

    This function maps the model's predicted token span back to the character
    indices of the original raw text.

    Args:
        text (str): The original raw tweet text.
        start_idx (int): The predicted index of the start token.
        end_idx (int): The predicted index of the end token.
        offsets (list of tuples): A list of (char_start, char_end) tuples
                                  corresponding to the tokens in the sequence.

    Returns:
        str: The extracted substring from the original text.
    """
    # If the start index is after the end index, the span is invalid.
    # We return the original text as a fallback, or an empty string could be used.
    # Given the task nature, returning the text is a reasonable fail-safe.
    if start_idx > end_idx:
        return text

    # Ensure indices are within the bounds of the offsets list
    if start_idx >= len(offsets) or end_idx >= len(offsets):
        return text

    # Retrieve the character positions from the offsets
    # offsets[i] is a tuple (start_char, end_char)
    selected_char_start = offsets[start_idx][0]
    selected_char_end = offsets[end_idx][1]

    # Extract the substring from the original text
    selected_text = text[selected_char_start:selected_char_end]

    # Strip leading/trailing whitespace that might be captured by the tokenizer
    # but isn't part of the semantic phrase.
    return selected_text.strip()
