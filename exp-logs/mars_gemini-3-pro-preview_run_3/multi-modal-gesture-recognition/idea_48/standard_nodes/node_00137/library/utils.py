import os
import random
import numpy as np
import torch
import nltk
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def levenshtein_distance(pred_seq, true_seq):
    """
    Calculates the Levenshtein edit distance between two sequences.

    Args:
        pred_seq (list): List of predicted tokens (e.g., gesture IDs).
        true_seq (list): List of ground truth tokens.

    Returns:
        int: The edit distance.
    """
    return nltk.edit_distance(pred_seq, true_seq)


def compute_edit_distance_score(predictions, ground_truths):
    """
    Computes the normalized Levenshtein distance score for a batch of predictions.
    Score = (Sum of Levenshtein Distances) / (Total Number of Ground Truth Gestures)

    Args:
        predictions (list of lists): Predicted sequences of gesture IDs.
        ground_truths (list of lists): Ground truth sequences of gesture IDs.

    Returns:
        float: The calculated error rate (lower is better).
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Mismatch in number of samples: predictions ({len(predictions)}) vs ground_truths ({len(ground_truths)})"
        )

    total_distance = 0
    total_gestures = 0

    for pred, truth in zip(predictions, ground_truths):
        # Calculate distance for this sequence
        dist = levenshtein_distance(pred, truth)
        total_distance += dist

        # Accumulate total number of gestures in ground truth
        total_gestures += len(truth)

    if total_gestures == 0:
        return 0.0

    return total_distance / total_gestures
