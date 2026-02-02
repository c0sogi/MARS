import os
import random
import numpy as np
import torch
import nltk


def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_levenshtein(prediction: list, target: list) -> int:
    """
    Computes the Levenshtein edit distance between two sequences.

    Args:
        prediction (list): The predicted sequence of gesture IDs.
        target (list): The ground truth sequence of gesture IDs.

    Returns:
        int: The edit distance.
    """
    return nltk.edit_distance(prediction, target)


def compute_normalized_levenshtein_score(predictions: dict, targets: dict) -> float:
    """
    Computes the normalized Levenshtein score (Error Rate) for the dataset.

    Score = Sum(Levenshtein(pred, target)) / Sum(len(target))

    Args:
        predictions (dict): Dictionary mapping sample_id to predicted sequence (list of IDs).
        targets (dict): Dictionary mapping sample_id to ground truth sequence (list of IDs).

    Returns:
        float: The normalized score.
    """
    total_distance = 0
    total_length = 0

    # Iterate over all samples in the targets
    for sample_id, target_seq in targets.items():
        # Get corresponding prediction, default to empty list if missing
        pred_seq = predictions.get(sample_id, [])

        # Compute distance
        dist = compute_levenshtein(pred_seq, target_seq)
        total_distance += dist

        # Accumulate target length
        total_length += len(target_seq)

    if total_length == 0:
        return 0.0

    return total_distance / total_length


def decode_predictions_to_sequence(frame_predictions: list) -> list:
    """
    Decodes frame-wise class predictions into a sequence of gesture IDs using
    Run-Length Encoding (RLE) logic to collapse consecutive identical labels.

    This corresponds to the 'Decoding' strategy in the AKC-IRN idea.
    It collapses consecutive duplicates and filters out background classes
    (assuming valid gestures are 1-20).

    Args:
        frame_predictions (list or np.array): List of frame-wise class IDs.

    Returns:
        list: Ordered list of recognized gesture IDs.
    """
    if len(frame_predictions) == 0:
        return []

    # 1. Collapse consecutive duplicates
    collapsed = [frame_predictions[0]]
    for i in range(1, len(frame_predictions)):
        if frame_predictions[i] != frame_predictions[i - 1]:
            collapsed.append(frame_predictions[i])

    # 2. Filter out background class.
    # Labels are 1-20. We filter out anything else (e.g., 0).
    final_sequence = [int(x) for x in collapsed if 1 <= x <= 20]

    return final_sequence
