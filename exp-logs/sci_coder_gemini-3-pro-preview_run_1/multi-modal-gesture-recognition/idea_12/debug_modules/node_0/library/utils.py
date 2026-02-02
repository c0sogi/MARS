import numpy as np
import torch
from itertools import groupby
from library.config import Config, seed_everything


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Uses the centralized seed_everything function from config.
    """
    seed_everything(seed)


def levenshtein_distance(pred_seq, target_seq):
    """
    Computes the Levenshtein edit distance between two sequences of integers.

    Args:
        pred_seq (list[int]): The predicted sequence of gesture IDs.
        target_seq (list[int]): The ground truth sequence of gesture IDs.

    Returns:
        int: The edit distance (insertions + deletions + substitutions).
    """
    n = len(pred_seq)
    m = len(target_seq)

    # Initialize DP table
    # dp[i][j] stores distance between pred_seq[:i] and target_seq[:j]
    dp = np.zeros((n + 1, m + 1), dtype=int)

    # Base cases: transforming to/from empty string
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    # Fill table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if pred_seq[i - 1] == target_seq[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Deletion
                dp[i][j - 1] + 1,  # Insertion
                dp[i - 1][j - 1] + cost,  # Substitution
            )

    return dp[n][m]


def compute_levenshtein(predictions, targets):
    """
    Computes the Levenshtein Error Rate over a batch or dataset.
    Metric = Sum(Levenshtein Distances) / Sum(Ground Truth Lengths)

    Args:
        predictions (list[list[int]]): List of predicted gesture sequences.
        targets (list[list[int]]): List of ground truth gesture sequences.

    Returns:
        float: The calculated error rate.
    """
    total_distance = 0
    total_target_length = 0

    for p, t in zip(predictions, targets):
        dist = levenshtein_distance(p, t)
        total_distance += dist
        total_target_length += len(t)

    # Avoid division by zero
    if total_target_length == 0:
        return 0.0 if total_distance == 0 else float("inf")

    return total_distance / total_target_length


def median_filter_1d(signal, window_size=Config.MEDIAN_FILTER_WINDOW):
    """
    Applies a median filter to a 1D sequence using NumPy.
    Used to smooth frame-wise predictions.

    Args:
        signal (list or np.ndarray): Input 1D signal (class indices).
        window_size (int): Size of the smoothing window.

    Returns:
        np.ndarray: Smoothed signal.
    """
    signal = np.array(signal)
    if window_size % 2 == 0:
        window_size += 1

    k = window_size // 2
    n = len(signal)
    output = np.zeros_like(signal)

    # Pad signal at boundaries with edge values
    padded = np.pad(signal, (k, k), mode="edge")

    # Sliding window median
    for i in range(n):
        window = padded[i : i + window_size]
        output[i] = np.median(window)

    return output.astype(int)


def decode_predictions(frame_predictions):
    """
    Decodes frame-wise class indices into a sequence of gesture IDs.

    Pipeline:
    1. Apply Median Filter to smooth noise.
    2. Run-Length Encoding (RLE) to group consecutive frames.
    3. Filter out background class (0).
    4. Filter out segments shorter than MIN_GESTURE_LENGTH.

    Args:
        frame_predictions (np.ndarray or list): 1D array of frame-wise class indices.

    Returns:
        list[int]: Ordered list of recognized gesture IDs.
    """
    # 1. Median Filter
    smoothed = median_filter_1d(frame_predictions)

    # 2. RLE
    # groupby returns (key, group_iterator)
    grouped = [(k, sum(1 for _ in g)) for k, g in groupby(smoothed)]

    decoded_sequence = []

    for class_id, length in grouped:
        class_id = int(class_id)

        # 3. Filter Background (0 is reserved for background)
        if class_id == 0:
            continue

        # 4. Filter Short Segments
        if length < Config.MIN_GESTURE_LENGTH:
            continue

        decoded_sequence.append(class_id)

    return decoded_sequence


def batch_decode(logits, lengths=None):
    """
    Decodes a batch of logits into gesture sequences.

    Args:
        logits (torch.Tensor): Shape (Batch, Time, Classes).
        lengths (torch.Tensor, optional): Valid lengths for each sequence in the batch.
                                          Used to mask out padding.

    Returns:
        list[list[int]]: Predicted sequences for each sample in the batch.
    """
    # Get frame-wise class predictions
    preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    results = []

    for i, seq in enumerate(preds):
        if lengths is not None:
            valid_len = lengths[i]
            valid_seq = seq[:valid_len]
        else:
            valid_seq = seq

        decoded = decode_predictions(valid_seq)
        results.append(decoded)

    return results
