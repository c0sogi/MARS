import os
import random
import numpy as np
import torch
import nltk
from itertools import groupby
from library import config


def set_seed(seed=config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def collapse_predictions(frame_predictions):
    """
    Converts a sequence of frame-wise predictions into a list of gesture IDs
    according to the competition format.

    Logic:
    1. Collapses consecutive duplicate labels (e.g., [1, 1, 2] -> [1, 2]).
    2. Removes the background class (ID 0).

    Args:
        frame_predictions (list or np.array): Sequence of predicted class indices.

    Returns:
        list: Ordered list of recognized gesture IDs (integers).
    """
    # Group consecutive elements to remove duplicates
    collapsed_sequence = [key for key, _ in groupby(frame_predictions)]

    # Filter out background class (0)
    gesture_sequence = [g for g in collapsed_sequence if g != 0]

    return gesture_sequence


def compute_levenshtein_score(predictions, targets):
    """
    Computes the Levenshtein error rate metric.

    Metric Definition:
    Sum of Levenshtein distances between predicted and target sequences,
    divided by the total number of gestures in the target sequences.

    Args:
        predictions (list of lists): Predicted gesture sequences.
        targets (list of lists): Ground truth gesture sequences.

    Returns:
        float: The calculated error rate.
    """
    total_distance = 0
    total_target_gestures = 0

    for pred_seq, target_seq in zip(predictions, targets):
        # Compute edit distance for the pair
        # nltk.edit_distance calculates the Levenshtein distance
        dist = nltk.edit_distance(pred_seq, target_seq)

        total_distance += dist
        total_target_gestures += len(target_seq)

    # Avoid division by zero if dataset is empty or has no gestures
    if total_target_gestures == 0:
        return 0.0

    return total_distance / total_target_gestures
