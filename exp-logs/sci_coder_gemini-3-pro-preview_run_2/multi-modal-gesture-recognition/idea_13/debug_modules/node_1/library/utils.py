import os
import random
import numpy as np
import torch
import nltk


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    Enforces deterministic behavior in cuDNN backends.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    """
    Returns the appropriate PyTorch device (cuda or cpu).

    Returns:
        torch.device: The available device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def compute_levenshtein(predicted_seq, target_seq):
    """
    Computes the Levenshtein edit distance between two sequences of gesture IDs.

    Args:
        predicted_seq (list or np.ndarray): Sequence of predicted labels.
        target_seq (list or np.ndarray): Sequence of ground truth labels.

    Returns:
        int: The Levenshtein distance (number of insertions, deletions, or substitutions).
    """
    # Ensure inputs are lists for nltk
    if isinstance(predicted_seq, np.ndarray):
        predicted_seq = predicted_seq.tolist()
    if isinstance(target_seq, np.ndarray):
        target_seq = target_seq.tolist()

    # NLTK's edit_distance handles lists of integers correctly
    return nltk.edit_distance(predicted_seq, target_seq)


def compute_score(predictions, ground_truths):
    """
    Computes the competition metric: Sum of Levenshtein Distances / Total Number of True Gestures.
    This score is analogous to an error rate and can exceed 1.0.

    Args:
        predictions (list of lists): List of predicted label sequences.
        ground_truths (list of lists): List of ground truth label sequences.

    Returns:
        float: The computed error rate.
    """
    total_distance = 0
    total_gestures = 0

    for pred, true in zip(predictions, ground_truths):
        # Compute distance for this sample
        dist = compute_levenshtein(pred, true)
        total_distance += dist
        total_gestures += len(true)

    # Avoid division by zero
    if total_gestures == 0:
        return 0.0

    return total_distance / total_gestures
