import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Ensures deterministic behavior in CuDNN.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU, though we have one here

    # Ensure deterministic behavior
    # Note: This may impact performance but is required for full reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def apk(actual, predicted, k=5):
    """
    Computes the average precision at k for a single sample.

    Args:
        actual (list): The ground truth items.
        predicted (list): The predicted items (ordered).
        k (int): The cutoff for predictions.

    Returns:
        float: The average precision at k.
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=5):
    """
    Computes the mean average precision at k across all samples.

    Args:
        actual (list or list of lists): The ground truth labels.
            Can be a list of single labels (e.g., ['id1', 'id2']) or
            a list of lists (e.g., [['id1'], ['id2']]).
        predicted (list of lists): The predicted labels for each sample.
        k (int): The cutoff for predictions.

    Returns:
        float: The mean average precision at k.
    """
    # Normalize actual to be a list of lists if it's a list of single items (strings/scalars)
    if len(actual) > 0 and not isinstance(actual[0], (list, tuple, np.ndarray)):
        actual = [[a] for a in actual]

    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def map5(actual, predicted):
    """
    Wrapper for mapk with k=5, specifically for the competition metric.

    Args:
        actual (list): The ground truth labels.
        predicted (list of lists): The predicted labels.

    Returns:
        float: The MAP@5 score.
    """
    return mapk(actual, predicted, k=5)
