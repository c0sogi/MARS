import random
import numpy as np
import torch
import nltk


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Returns the appropriate torch device (CUDA if available, else CPU).

    Returns:
        torch.device: The selected device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_levenshtein(predictions, targets):
    """
    Computes the Levenshtein distance metric as defined in the task description.

    Metric = (Sum of Levenshtein distances) / (Total number of ground truth gestures)

    Args:
        predictions (list of list of int): List of predicted gesture sequences.
        targets (list of list of int): List of ground truth gesture sequences.

    Returns:
        float: The computed error rate (lower is better).
    """
    total_distance = 0
    total_target_length = 0

    for pred_seq, target_seq in zip(predictions, targets):
        # Ensure inputs are lists and handle None
        p = list(pred_seq) if pred_seq is not None else []
        t = list(target_seq) if target_seq is not None else []

        # Calculate Levenshtein distance between two sequences
        # nltk.edit_distance works efficiently on lists of integers
        dist = nltk.edit_distance(p, t)

        total_distance += dist
        total_target_length += len(t)

    if total_target_length == 0:
        return 0.0

    return total_distance / total_target_length
