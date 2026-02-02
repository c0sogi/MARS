import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def jaccard(str1, str2):
    """
    Calculates the Word-level Jaccard score between two strings.

    Args:
        str1 (str): The first string (e.g., ground truth).
        str2 (str): The second string (e.g., prediction).

    Returns:
        float: The Jaccard similarity score [0.0, 1.0].
    """
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    denominator = len(a) + len(b) - len(c)
    if denominator == 0:
        return 0.0
    return float(len(c)) / denominator


def decode_bio_spans(context, tags, offsets):
    """
    Decodes BIO tags and token offsets into the extracted answer string.

    Assumes the following tag mapping:
    0: O (Outside)
    1: B (Beginning of Answer)
    2: I (Inside of Answer)

    Args:
        context (str): The original context text.
        tags (list[int] or np.ndarray): Sequence of predicted tags.
        offsets (list[tuple]): Sequence of (start, end) character offsets for each token.

    Returns:
        str: The extracted answer text. Returns an empty string if no valid span is found.
    """
    spans = []
    current_span = None

    # Iterate through tags and offsets to identify spans
    for tag, (start, end) in zip(tags, offsets):
        # Skip special tokens or invalid offsets (where start >= end)
        if start >= end:
            continue

        if tag == 1:  # B-ANS
            if current_span:
                spans.append(current_span)
            current_span = [start, end]

        elif tag == 2:  # I-ANS
            if current_span:
                # Extend the current span
                current_span[1] = end
            else:
                # Found I without B. Treat as start of a new span for robustness.
                current_span = [start, end]

        else:  # O
            if current_span:
                spans.append(current_span)
                current_span = None

    # Append the last span if it exists
    if current_span:
        spans.append(current_span)

    if not spans:
        return ""

    # In this implementation, we return the text of the first detected span.
    # Advanced logic (like confidence scoring) is handled in the inference loop.
    best_span = spans[0]

    # Extract the substring from the original context using character offsets
    return context[best_span[0] : best_span[1]]
