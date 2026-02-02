import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def apk(actual, predicted, k=5):
    """
    Computes the average precision at k.

    This function computes the average prescision at k between two lists of
    items.

    Args:
        actual : list
            A list of elements that are to be predicted (order doesn't matter)
        predicted : list
            A list of predicted elements (order does matter)
        k : int, optional
            The maximum number of predicted elements

    Returns:
        score : double
            The average precision at k over the input lists
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
    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists
    of lists of items.

    Args:
        actual : list
            A list of lists of elements that are to be predicted
            (order doesn't matter in the inner lists)
        predicted : list
            A list of lists of predicted elements
            (order matters in the inner lists)
        k : int, optional
            The maximum number of predicted elements

    Returns:
        score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])
