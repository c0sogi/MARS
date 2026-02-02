import numpy as np
from scipy.ndimage import median_filter as scipy_median_filter
from library.config import DEBUG


def compute_levenshtein(seq1, seq2):
    """
    Computes the Levenshtein edit distance between two sequences.

    Args:
        seq1 (list or np.array): First sequence of items (e.g., predicted gesture IDs).
        seq2 (list or np.array): Second sequence of items (e.g., ground truth gesture IDs).

    Returns:
        int: The edit distance.
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y), dtype=int)

    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1, matrix[x - 1, y - 1], matrix[x, y - 1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1, matrix[x - 1, y - 1] + 1, matrix[x, y - 1] + 1
                )
    return matrix[size_x - 1, size_y - 1]


def calculate_levenshtein_accuracy(preds, targets):
    """
    Calculates the competition metric: Sum of Levenshtein distances divided by
    total number of ground truth gestures.

    Args:
        preds (list of lists): Predicted sequences of gesture IDs.
        targets (list of lists): Ground truth sequences of gesture IDs.

    Returns:
        float: The error rate (lower is better).
    """
    total_distance = 0
    total_length = 0

    for p, t in zip(preds, targets):
        dist = compute_levenshtein(p, t)
        total_distance += dist
        total_length += len(t)

    if total_length == 0:
        return 0.0

    return total_distance / total_length


def median_filter(probs, window_size=5):
    """
    Applies a median filter to the probability outputs along the temporal axis.

    Args:
        probs (np.ndarray): Array of shape (Time, Classes).
        window_size (int): Size of the smoothing window.

    Returns:
        np.ndarray: Smoothed probabilities.
    """
    # Apply median filter along the first axis (Time)
    # footprint defines the window: (window_size, 1) means window over time, independent per class
    return scipy_median_filter(probs, size=(window_size, 1), mode="nearest")


def rle_decode(predictions, bg_class=0, min_len=5):
    """
    Decodes frame-wise predictions into a sequence of gesture IDs using Run-Length Encoding.
    Filters out background class and short segments.

    Args:
        predictions (np.ndarray): Frame-wise class indices of shape (Time,).
        bg_class (int): The class index representing background/no-gesture.
        min_len (int): Minimum duration (in frames) for a gesture to be considered valid.

    Returns:
        list: Ordered list of recognized gesture IDs.
    """
    if len(predictions) == 0:
        return []

    decoded_sequence = []

    # Run-Length Encoding logic
    current_class = predictions[0]
    current_len = 1

    for i in range(1, len(predictions)):
        cls = predictions[i]
        if cls == current_class:
            current_len += 1
        else:
            # End of previous run
            if current_class != bg_class and current_len >= min_len:
                decoded_sequence.append(int(current_class))

            # Start new run
            current_class = cls
            current_len = 1

    # Handle the last run
    if current_class != bg_class and current_len >= min_len:
        decoded_sequence.append(int(current_class))

    # Post-processing: Collapse consecutive duplicates if they exist after filtering?
    # The prompt implies standard RLE on the raw stream handles this, but if noise caused
    # A A A B B A A A (where B is short), filtering B might leave A A A A A A.
    # However, standard practice for this dataset is usually sequential extraction.
    # We simply return the list of valid segments found in order.

    return decoded_sequence


def post_process_output(probs, window_size=5, min_len=5, bg_class=0):
    """
    Wrapper function to apply smoothing and decoding.

    Args:
        probs (np.ndarray): Raw probabilities from model (Time, Classes).
        window_size (int): Median filter window size.
        min_len (int): Minimum segment length.
        bg_class (int): Background class index.

    Returns:
        list: Decoded sequence of gesture IDs.
    """
    # 1. Smooth
    smoothed_probs = median_filter(probs, window_size=window_size)

    # 2. Argmax
    preds = np.argmax(smoothed_probs, axis=1)

    # 3. Decode
    sequence = rle_decode(preds, bg_class=bg_class, min_len=min_len)

    return sequence
