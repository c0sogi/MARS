import os
import random
import numpy as np
import torch
from nltk import edit_distance


def set_seed(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_levenshtein(preds, targets):
    """
    Computes the normalized Levenshtein distance (Error Rate).

    Metric = (Sum of Levenshtein distances) / (Total number of gestures in ground truth)

    Args:
        preds (list of list of int): Predicted gesture sequences.
        targets (list of list of int): Ground truth gesture sequences.

    Returns:
        float: The calculated error rate.
    """
    total_distance = 0
    total_target_length = 0

    for p, t in zip(preds, targets):
        # Ensure inputs are lists
        p_seq = list(p) if p is not None else []
        t_seq = list(t) if t is not None else []

        # Calculate Levenshtein distance for this sequence
        dist = edit_distance(p_seq, t_seq)
        total_distance += dist

        # Accumulate total length of ground truth gestures
        total_target_length += len(t_seq)

    if total_target_length == 0:
        # Avoid division by zero; if there are no targets, error is 0 if preds are empty, else infinite/undefined.
        # Returning 0.0 if both are empty, else 1.0 as a penalty if preds exist but targets don't.
        if total_distance == 0:
            return 0.0
        else:
            return 1.0

    return total_distance / total_target_length
