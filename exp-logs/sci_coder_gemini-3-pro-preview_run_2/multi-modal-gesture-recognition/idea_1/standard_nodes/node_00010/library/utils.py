import numpy as np
import torch
import nltk
from scipy.ndimage import median_filter
from library.config import Config, set_seed


def compute_levenshtein(predicted_seqs, truth_seqs):
    """
    Computes the Levenshtein distance metric for the batch or dataset.
    Metric = Sum(Levenshtein Distances) / Total Number of Ground Truth Gestures.

    Args:
        predicted_seqs (list of list of int): List of predicted gesture sequences.
        truth_seqs (list of list of int): List of ground truth gesture sequences.

    Returns:
        float: The computed error rate (Levenshtein distance normalized by total truth length).
    """
    if len(predicted_seqs) != len(truth_seqs):
        raise ValueError(
            f"Mismatch in number of sequences: {len(predicted_seqs)} vs {len(truth_seqs)}"
        )

    total_distance = 0
    total_truth_length = 0

    for pred, truth in zip(predicted_seqs, truth_seqs):
        # nltk.edit_distance computes the Levenshtein distance
        dist = nltk.edit_distance(pred, truth)
        total_distance += dist
        total_truth_length += len(truth)

    # Avoid division by zero if the ground truth is empty (unlikely in valid sets)
    if total_truth_length == 0:
        return 0.0 if total_distance == 0 else float("inf")

    return total_distance / total_truth_length


def decode_predictions(logits):
    """
    Decodes frame-wise logits into gesture sequences using a greedy strategy.

    Strategy:
    1. Argmax: Select class with highest probability for each frame.
    2. Collapse Repeats: Merge consecutive identical classes.
    3. Remove Background: Filter out the background class (index 0).

    Args:
        logits (torch.Tensor or np.ndarray): Model outputs of shape (Batch, Time, Classes)
                                             or (Time, Classes).

    Returns:
        list of list of int: Decoded sequences of gesture IDs.
    """
    # Convert Tensor to Numpy if necessary
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()

    # Ensure input is 3D (Batch, Time, Classes)
    if logits.ndim == 2:
        logits = logits[np.newaxis, :, :]

    # Step 1: Argmax along the class dimension
    # Shape becomes (Batch, Time)
    raw_predictions = np.argmax(logits, axis=-1)

    decoded_sequences = []

    for seq in raw_predictions:
        if len(seq) == 0:
            decoded_sequences.append([])
            continue

        # Apply Median Filter to smooth predictions (Cite solution_lesson_node_00006)
        # Use nearest padding to preserve boundaries (Cite solution_lesson_node_00009)
        if Config.MEDIAN_FILTER_KERNEL > 1:
            seq = median_filter(
                seq, size=Config.MEDIAN_FILTER_KERNEL, mode="nearest"
            ).astype(int)

        # Step 2: Collapse consecutive repeats
        collapsed_seq = [seq[0]]
        for i in range(1, len(seq)):
            if seq[i] != seq[i - 1]:
                collapsed_seq.append(seq[i])

        # Step 3: Remove background class (0)
        # We cast to int to ensure standard Python types
        final_seq = [int(label) for label in collapsed_seq if label != 0]

        decoded_sequences.append(final_seq)

    return decoded_sequences
