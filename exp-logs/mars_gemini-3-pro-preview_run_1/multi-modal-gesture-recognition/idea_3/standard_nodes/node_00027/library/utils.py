import os
import random
import numpy as np
import torch
import scipy.signal
import nltk
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def levenshtein_distance(pred_seq, target_seq):
    """
    Computes the Levenshtein edit distance between two sequences of gesture IDs.

    Args:
        pred_seq (list): List of predicted gesture IDs (integers).
        target_seq (list): List of ground truth gesture IDs (integers).

    Returns:
        int: The edit distance.
    """
    return nltk.edit_distance(pred_seq, target_seq)


def compute_normalized_levenshtein(predictions, ground_truths):
    """
    Computes the global normalized Levenshtein score (Error Rate) for a dataset.
    Score = Sum(Levenshtein Distances) / Sum(Ground Truth Lengths).

    Args:
        predictions (list of lists): Predicted sequences for the dataset.
        ground_truths (list of lists): Ground truth sequences for the dataset.

    Returns:
        float: The aggregate error rate.
    """
    total_dist = 0
    total_len = 0

    for p, t in zip(predictions, ground_truths):
        # Ensure inputs are lists
        p_list = list(p) if isinstance(p, (np.ndarray, torch.Tensor)) else p
        t_list = list(t) if isinstance(t, (np.ndarray, torch.Tensor)) else t

        dist = levenshtein_distance(p_list, t_list)
        total_dist += dist
        total_len += len(t_list)

    if total_len == 0:
        return 0.0

    return total_dist / total_len


def _rle_encode(sequence):
    """
    Helper function for Run-Length Encoding.

    Args:
        sequence (np.ndarray): 1D array of labels.

    Returns:
        list of tuples: (label, length)
    """
    if len(sequence) == 0:
        return []

    encoded = []
    curr_val = sequence[0]
    curr_len = 1

    for i in range(1, len(sequence)):
        if sequence[i] == curr_val:
            curr_len += 1
        else:
            encoded.append((curr_val, curr_len))
            curr_val = sequence[i]
            curr_len = 1
    encoded.append((curr_val, curr_len))

    return encoded


def _decode_single_sequence(logits):
    """
    Helper to decode a single sequence of logits into a gesture list.
    """
    # 1. Argmax to get class indices
    raw_preds = np.argmax(logits, axis=-1)  # Shape (T,)

    # 2. Median Filter to smooth predictions
    # Kernel size must be odd
    k = Config.MEDIAN_FILTER_KERNEL
    if k % 2 == 0:
        k += 1

    # medfilt may return floats, cast back to int
    smoothed_preds = scipy.signal.medfilt(raw_preds, kernel_size=k).astype(int)

    # 3. Run-Length Encoding
    segments = _rle_encode(smoothed_preds)

    # 4. Filter segments
    final_sequence = []
    for label, length in segments:
        # Filter out Background class
        if label == Config.BACKGROUND_CLASS_IDX:
            continue

        # Filter out short gestures
        if length < Config.MIN_GESTURE_LENGTH:
            continue

        final_sequence.append(int(label))

    return final_sequence


def decode_predictions(frame_logits):
    """
    Decodes frame-wise logits into a sequence of gesture IDs.
    Applies Median Filtering, RLE, and heuristic filtering (background removal, min duration).

    Args:
        frame_logits (np.ndarray or torch.Tensor):
            Shape (T, C) for a single sequence or (B, T, C) for a batch.

    Returns:
        list:
            - If input is (T, C): A list of integer gesture IDs.
            - If input is (B, T, C): A list of lists of integer gesture IDs.
    """
    # Convert Tensor to Numpy
    if isinstance(frame_logits, torch.Tensor):
        frame_logits = frame_logits.detach().cpu().numpy()

    # Handle Batch vs Single Sequence
    if frame_logits.ndim == 3:
        # Batch mode
        batch_preds = []
        for i in range(frame_logits.shape[0]):
            batch_preds.append(_decode_single_sequence(frame_logits[i]))
        return batch_preds

    elif frame_logits.ndim == 2:
        # Single sequence mode
        return _decode_single_sequence(frame_logits)

    else:
        raise ValueError(
            f"Invalid shape for frame_logits: {frame_logits.shape}. Expected 2D (T, C) or 3D (B, T, C)."
        )
