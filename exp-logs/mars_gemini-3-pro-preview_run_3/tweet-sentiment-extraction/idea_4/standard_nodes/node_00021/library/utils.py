import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def jaccard(str1, str2):
    """
    Calculates the word-level Jaccard similarity score between two strings.

    Args:
        str1 (str): The first string (e.g., predicted text).
        str2 (str): The second string (e.g., ground truth text).

    Returns:
        float: The Jaccard score between 0.0 and 1.0.
    """
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())

    # Handle case where both sets are empty
    if len(a) == 0 and len(b) == 0:
        return 0.5

    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def get_selected_text(text, start_probs, end_probs, sentiment, offsets):
    """
    Decodes the selected text from the model probabilities using Summation Decoding.

    Args:
        text (str): The original tweet text.
        start_probs (np.ndarray): 1D array of start token probabilities.
        end_probs (np.ndarray): 1D array of end token probabilities.
        sentiment (str): The sentiment of the tweet.
        offsets (list or np.ndarray): List of (char_start, char_end) tuples for each token.

    Returns:
        str: The extracted text span.
    """
    # Rule 1: Deterministic rule for neutral sentiment
    # Neutral tweets almost always have selected_text == text
    if sentiment == "neutral":
        return text

    # Rule 2: Summation Decoding for positive/negative
    # We want to find indices (i, j) such that i <= j and start_probs[i] + end_probs[j] is maximized.

    # Ensure inputs are numpy arrays
    start_probs = np.asarray(start_probs)
    end_probs = np.asarray(end_probs)

    seq_len = len(start_probs)
    if seq_len == 0:
        return text

    # Create a matrix of sums: M[i, j] = start_probs[i] + end_probs[j]
    # Shape: (seq_len, seq_len)
    score_matrix = np.add.outer(start_probs, end_probs)

    # Mask out the lower triangle (where end_index < start_index)
    # np.triu returns the upper triangle (including diagonal k=0)
    # We set invalid positions to negative infinity
    mask = np.triu(np.ones((seq_len, seq_len)), k=0)
    score_matrix = score_matrix * mask + (1 - mask) * -1e9

    # Find the indices of the maximum score in the flattened matrix
    flat_idx = np.argmax(score_matrix)

    # Convert flat index back to (start_idx, end_idx)
    start_idx, end_idx = np.unravel_index(flat_idx, (seq_len, seq_len))

    # Map token indices to character offsets to extract the substring
    try:
        start_char = offsets[start_idx][0]
        end_char = offsets[end_idx][1]

        # Extract the text using the character boundaries
        selected_text = text[start_char:end_char]
        return selected_text
    except IndexError:
        # Fallback if indices are out of bounds (unlikely with correct logic)
        return text


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
