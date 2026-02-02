import random
import os
import numpy as np
import torch
from nltk import edit_distance


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def levenshtein_score(predictions, targets):
    """
    Calculates the Levenshtein score (error rate) for the given predictions and targets.

    The score is calculated as the sum of Levenshtein distances between predicted and
    target sequences, divided by the total number of gestures in the target sequences.

    Args:
        predictions (list of list of int): List of predicted gesture sequences.
        targets (list of list of int): List of ground truth gesture sequences.

    Returns:
        float: The computed Levenshtein score.
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"Predictions and targets must have the same length. Got {len(predictions)} and {len(targets)}."
        )

    total_distance = 0
    total_target_length = 0

    for pred_seq, target_seq in zip(predictions, targets):
        # Calculate Levenshtein distance for the pair
        # nltk.edit_distance works with lists of integers directly
        dist = edit_distance(pred_seq, target_seq)
        total_distance += dist
        total_target_length += len(target_seq)

    # Avoid division by zero if targets are empty
    if total_target_length == 0:
        return 0.0

    score = total_distance / total_target_length
    return score
