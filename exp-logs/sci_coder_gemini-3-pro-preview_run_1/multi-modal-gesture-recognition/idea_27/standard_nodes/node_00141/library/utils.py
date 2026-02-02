import os
import random
import numpy as np
import torch
import nltk
from scipy.ndimage import median_filter
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def decode_predictions(frame_probs):
    """
    Decodes frame-wise probabilities into a sequence of gesture labels.

    Applies:
    1. Argmax to get class indices.
    2. Median Filter (window=5) for smoothing.
    3. Run-Length Encoding (RLE) to extract segments.
    4. Filtering: Removes Background class (0) and segments < 5 frames.

    Args:
        frame_probs (np.ndarray): Array of shape (T, NumClasses) containing probabilities or logits.

    Returns:
        list: Ordered list of predicted gesture IDs (int).
    """
    # 1. Get frame labels
    raw_preds = np.argmax(frame_probs, axis=1)

    # 2. Apply Median Filter
    # Window size 5 as per "Idea" description
    smoothed_preds = median_filter(raw_preds, size=5, mode="nearest")

    # 3. Run-Length Encoding (RLE)
    segments = []
    if len(smoothed_preds) == 0:
        return []

    current_label = smoothed_preds[0]
    current_len = 1

    for label in smoothed_preds[1:]:
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
        # Filter out background class
        if label == Config.BACKGROUND_CLASS_ID:
            continue

        # Filter out short segments
        if length < 5:
            continue

        final_sequence.append(int(label))

    return final_sequence


def compute_levenshtein(predictions, targets):
    """
    Computes the Levenshtein error rate for a batch of predictions.

    Metric = Sum(Levenshtein Distance) / Sum(Ground Truth Length)

    Args:
        predictions (list of list): List of predicted gesture sequences.
        targets (list of list): List of ground truth gesture sequences.

    Returns:
        float: The calculated error rate.
    """
    total_distance = 0
    total_len = 0

    for pred_seq, target_seq in zip(predictions, targets):
        # Ensure inputs are lists of integers
        p_seq = list(pred_seq)
        t_seq = list(target_seq)

        # Calculate distance
        dist = nltk.edit_distance(p_seq, t_seq)
        total_distance += dist

        # Accumulate target length
        total_len += len(t_seq)

    if total_len == 0:
        return 0.0

    return total_distance / total_len
