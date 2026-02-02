import os
import random
import numpy as np
import torch
import scipy.ndimage
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def _levenshtein_distance(seq1, seq2):
    """
    Calculates the Levenshtein distance between two sequences.
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))

    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = matrix[x - 1, y - 1]
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,  # Deletion
                    matrix[x - 1, y - 1] + 1,  # Substitution
                    matrix[x, y - 1] + 1,  # Insertion
                )
    return matrix[size_x - 1, size_y - 1]


def compute_levenshtein_distance(predictions, ground_truths):
    """
    Computes the normalized Levenshtein distance metric.

    Args:
        predictions (list of list of int): Predicted gesture IDs for each sequence.
        ground_truths (list of list of int): Ground truth gesture IDs for each sequence.

    Returns:
        float: The total Levenshtein distance divided by the total number of ground truth gestures.
    """
    total_dist = 0
    total_len = 0

    for pred, true in zip(predictions, ground_truths):
        dist = _levenshtein_distance(pred, true)
        total_dist += dist
        total_len += len(true)

    if total_len == 0:
        return 0.0

    return total_dist / total_len


def decode_predictions(frame_logits, threshold=5, bg_class=Config.BACKGROUND_CLASS_ID):
    """
    Decodes frame-wise logits into a sequence of gesture IDs.

    Steps:
    1. Argmax to get class indices.
    2. Median filter to smooth noise.
    3. Run-Length Encoding to group contiguous frames.
    4. Filter out background class and short segments.

    Args:
        frame_logits (np.ndarray): Array of shape (T, NumClasses).
        threshold (int): Minimum duration (frames) to keep a gesture.
        bg_class (int): The class ID representing background/silence.

    Returns:
        list of int: The ordered list of recognized gesture IDs.
    """
    # 1. Get class indices
    if isinstance(frame_logits, torch.Tensor):
        frame_logits = frame_logits.detach().cpu().numpy()

    preds = np.argmax(frame_logits, axis=1)

    # 2. Median Filter
    # Use a window size of 5 as per strategy
    preds_smoothed = scipy.ndimage.median_filter(preds, size=5, mode="nearest")

    # 3. Run-Length Encoding & 4. Filtering
    decoded_sequence = []

    if len(preds_smoothed) == 0:
        return decoded_sequence

    # Iteration for RLE
    current_label = preds_smoothed[0]
    current_len = 1

    # Helper to append if valid
    def append_if_valid(label, length):
        if label != bg_class and length >= threshold:
            # Avoid repeating the same gesture if it was just added (optional,
            # but standard RLE implies distinct segments. If the user performs
            # gesture A, pauses (bg), then gesture A again, it should be A, A.
            # If the user performs A then A immediately without pause, the logic
            # below treats it as one long A block.
            # The prompt implies a sequence of gestures.
            decoded_sequence.append(int(label))

    for i in range(1, len(preds_smoothed)):
        label = preds_smoothed[i]
        if label == current_label:
            current_len += 1
        else:
            append_if_valid(current_label, current_len)
            current_label = label
            current_len = 1

    # Append the last segment
    append_if_valid(current_label, current_len)

    return decoded_sequence
