import os
import random
import numpy as np
import torch
from library.config import SEED, BACKGROUND_CLASS_ID


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def levenshtein_distance(hyp, ref):
    """
    Computes the Levenshtein edit distance between two sequences of labels.

    Args:
        hyp (list): The predicted sequence of gesture IDs.
        ref (list): The ground truth sequence of gesture IDs.

    Returns:
        int: The Levenshtein distance.
    """
    n = len(hyp)
    m = len(ref)

    # Optimization: ensure we iterate over the shorter sequence for inner loop if desired,
    # but standard DP matrix approach is O(N*M).
    if n < m:
        return levenshtein_distance(ref, hyp)

    if m == 0:
        return n

    # Initialize previous row (distance from empty string)
    previous_row = list(range(m + 1))

    for i, c1 in enumerate(hyp):
        current_row = [i + 1]
        for j, c2 in enumerate(ref):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (0 if c1 == c2 else 1)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def decode_predictions(frame_logits):
    """
    Decodes frame-wise logits into a clean sequence of gesture IDs.

    Pipeline:
    1. Argmax to get class indices.
    2. Median Filter (window=5) for temporal smoothing.
    3. Run-Length Encoding (RLE) to identify segments.
    4. Filter out Background class and segments shorter than 5 frames.

    Args:
        frame_logits (np.ndarray or torch.Tensor): Logits of shape (T, NumClasses).

    Returns:
        list: Ordered list of recognized gesture IDs.
    """
    # Ensure input is a numpy array
    if isinstance(frame_logits, torch.Tensor):
        frame_logits = frame_logits.detach().cpu().numpy()

    if frame_logits.shape[0] == 0:
        return []

    # 1. Get raw class indices
    predicted_indices = np.argmax(frame_logits, axis=-1)

    # 2. Median Filter (Window Size = 5)
    # Using numpy to avoid scipy dependency issues
    k = 5
    pad_size = k // 2
    # Pad with edge values to simulate 'nearest' mode
    padded = np.pad(predicted_indices, (pad_size, pad_size), mode="edge")
    # Create sliding windows
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=k)
    # Compute median (odd window size ensures integer result mathematically, but numpy returns float)
    smoothed_indices = np.median(windows, axis=1).astype(int)

    # 3. Run-Length Encoding (RLE)
    segments = []
    if len(smoothed_indices) > 0:
        current_label = smoothed_indices[0]
        current_len = 1

        for label in smoothed_indices[1:]:
            if label == current_label:
                current_len += 1
            else:
                segments.append((current_label, current_len))
                current_label = label
                current_len = 1
        segments.append((current_label, current_len))

    # 4. Filter segments
    final_sequence = []
    for label, length in segments:
        # Skip background class
        if label == BACKGROUND_CLASS_ID:
            continue

        # Skip short segments (noise)
        if length < 5:
            continue

        final_sequence.append(int(label))

    return final_sequence
