import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the seed for all random number generators to ensure reproducibility.
    Delegates to Config.seed_everything to maintain a single source of truth.
    """
    Config.seed_everything(seed)


def jaccard(str1, str2):
    """
    Calculates the word-level Jaccard score between two strings.

    Args:
        str1 (str): The prediction string.
        str2 (str): The ground truth string.

    Returns:
        float: The Jaccard similarity score (0.0 to 1.0).
    """
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())

    c = a.intersection(b)
    union_len = len(a) + len(b) - len(c)

    if union_len == 0:
        # If both sets are empty, we consider it a match (1.0)
        if len(a) == 0 and len(b) == 0:
            return 1.0
        return 0.0

    return float(len(c)) / union_len


def get_selected_text(text, start_idx, end_idx, offsets):
    """
    Reconstructs the selected text string from the original text using predicted
    start and end token indices and the tokenizer's offset mapping.

    Args:
        text (str): The original tweet text.
        start_idx (int): The predicted start token index.
        end_idx (int): The predicted end token index.
        offsets (list or np.array): List of (start_char, end_char) tuples for each token.

    Returns:
        str: The substring of text corresponding to the predicted token span.
    """
    # Basic validation: if start is after end, prediction is invalid.
    # In this task, returning the full text is a common fallback,
    # but strictly speaking, the caller should handle this.
    # We will return the text to be safe, or one could return an empty string.
    if start_idx > end_idx:
        return text

    # Handle out of bounds indices
    if start_idx < 0 or end_idx >= len(offsets):
        return text

    # Retrieve character indices from offsets
    # offsets[i] is a tuple (char_start, char_end)
    selected_char_start = offsets[start_idx][0]
    selected_char_end = offsets[end_idx][1]

    # Ensure indices are valid for the string (clamping)
    if selected_char_start < 0:
        selected_char_start = 0
    if selected_char_end > len(text):
        selected_char_end = len(text)

    return text[selected_char_start:selected_char_end]


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking metrics (loss, jaccard) during training/validation.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
