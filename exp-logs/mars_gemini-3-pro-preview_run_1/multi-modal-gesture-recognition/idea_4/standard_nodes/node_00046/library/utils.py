import numpy as np
import torch
import scipy.signal
from library.config import Config


def set_seed(seed=None):
    """
    Sets the random seed for reproducibility using the Config class.

    Args:
        seed (int, optional): The seed to set. If None, uses Config.SEED.
    """
    Config.set_seed(seed)


def compute_levenshtein(seq1, seq2):
    """
    Computes the Levenshtein distance between two sequences of gesture IDs.
    This metric counts the minimum number of single-element edits (insertions,
    deletions, or substitutions) required to change one sequence into the other.

    Args:
        seq1 (list or np.array): First sequence (e.g., predicted).
        seq2 (list or np.array): Second sequence (e.g., ground truth).

    Returns:
        float: The Levenshtein distance.
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
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,  # Deletion
                    matrix[x - 1, y - 1],  # Match
                    matrix[x, y - 1] + 1,  # Insertion
                )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,  # Deletion
                    matrix[x - 1, y - 1] + 1,  # Substitution
                    matrix[x, y - 1] + 1,  # Insertion
                )
    return matrix[size_x - 1, size_y - 1]


def decode_predictions(frame_probs):
    """
    Decodes frame-wise probabilities into a sequence of gesture IDs.

    Pipeline:
    1. Argmax to obtain frame-level labels.
    2. Median Filter (window size 5) to smooth noise.
    3. Run-Length Encoding (RLE) to group contiguous segments.
    4. Filtering: Removes background class (0) and segments shorter than 5 frames.

    Args:
        frame_probs (np.ndarray or torch.Tensor): Shape (T, NumClasses).

    Returns:
        list: Ordered list of predicted gesture IDs (integers).
    """
    # Convert to numpy if tensor
    if isinstance(frame_probs, torch.Tensor):
        frame_probs = frame_probs.detach().cpu().numpy()

    # 1. Argmax to get raw labels
    raw_labels = np.argmax(frame_probs, axis=1)

    # 2. Median Filter (window size 5)
    # kernel_size must be odd. Using 5 as per specification.
    filtered_labels = scipy.signal.medfilt(raw_labels, kernel_size=5).astype(int)

    # 3. Run-Length Encoding & Filtering
    predicted_gestures = []

    if len(filtered_labels) == 0:
        return predicted_gestures

    current_label = filtered_labels[0]
    current_len = 1

    # Helper to check validity and append
    def _process_segment(lbl, length):
        # Filter out background (0) and segments shorter than 5 frames
        if lbl != 0 and length >= 5:
            predicted_gestures.append(int(lbl))

    for i in range(1, len(filtered_labels)):
        lbl = filtered_labels[i]
        if lbl == current_label:
            current_len += 1
        else:
            _process_segment(current_label, current_len)
            current_label = lbl
            current_len = 1

    # Process the final segment
    _process_segment(current_label, current_len)

    return predicted_gestures
