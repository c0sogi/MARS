import numpy as np
from library.config import seed_everything


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking loss and metrics during training.
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


def jaccard(str1, str2):
    """
    Calculates the Word-level Jaccard score between two strings.
    The score is the intersection over union of the set of words.

    Args:
        str1 (str): The first string (e.g., ground truth).
        str2 (str): The second string (e.g., prediction).

    Returns:
        float: The Jaccard similarity coefficient (0.0 to 1.0).
    """
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())
    c = a.intersection(b)
    return (
        float(len(c)) / (len(a) + len(b) - len(c))
        if (len(a) + len(b) - len(c)) > 0
        else 0.0
    )


def normalize_text(text):
    """
    Normalizes text by collapsing multiple whitespaces into a single space.
    This implements the 'Normalize-First' protocol to ensure alignment between
    raw text and tokenizer offsets.

    Args:
        text (str): The input text.

    Returns:
        str: The normalized text.
    """
    return " ".join(str(text).split())


def get_soft_targets(seq_len, idx, sigma=1.0):
    """
    Generates Gaussian-smoothed targets for a specific index.
    Returns a probability distribution over the sequence length that sums to 1.

    Args:
        seq_len (int): The length of the sequence.
        idx (int): The center index (mean) of the Gaussian.
        sigma (float): The standard deviation of the Gaussian.

    Returns:
        np.ndarray: A 1D array of probabilities summing to 1.
    """
    x = np.arange(seq_len)
    # Gaussian function: exp(-0.5 * ((x - mu) / sigma)^2)
    logits = np.exp(-0.5 * ((x - idx) / sigma) ** 2)
    # Normalize to create a valid probability distribution
    return logits / logits.sum()


def get_best_start_end_idxs(start_logits, end_logits):
    """
    Finds the optimal start and end indices maximizing the sum of logits
    subject to start_idx <= end_idx.

    Args:
        start_logits (np.ndarray): Array of start logits (shape: seq_len).
        end_logits (np.ndarray): Array of end logits (shape: seq_len).

    Returns:
        tuple: (best_start_idx, best_end_idx)
    """
    # Create a matrix of sums: score[i, j] = start[i] + end[j]
    # start_logits: (N,) -> (N, 1)
    # end_logits: (N,) -> (1, N)
    score_matrix = start_logits[:, None] + end_logits[None, :]

    # Mask out invalid predictions where end < start (lower triangle)
    # np.triu returns the upper triangle (including diagonal) as 1s
    valid_mask = np.triu(np.ones_like(score_matrix))

    # Set invalid positions to negative infinity so they are not selected
    score_matrix = np.where(valid_mask == 1, score_matrix, -np.inf)

    # Find the flat index of the maximum value
    best_idx_flat = np.argmax(score_matrix)

    # Convert flat index back to (row, col) -> (start, end)
    best_start, best_end = np.unravel_index(best_idx_flat, score_matrix.shape)

    return best_start, best_end
