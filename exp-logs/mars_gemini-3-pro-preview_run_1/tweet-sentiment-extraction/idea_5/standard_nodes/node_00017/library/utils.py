import os
import random
import numpy as np
import torch
from library.config import set_seed


def seed_everything(seed=42):
    """
    Seeds all random number generators for reproducibility.
    Wraps the set_seed function from library.config to ensure consistent behavior across modules.

    Args:
        seed (int): The seed value to use.
    """
    set_seed(seed)


def jaccard(str1, str2):
    """
    Calculates the Word-level Jaccard score between two strings.
    The Jaccard score is the intersection over union of the set of words in the strings.

    Args:
        str1 (str): The first string (e.g., predicted text).
        str2 (str): The second string (e.g., ground truth text).

    Returns:
        float: The Jaccard score between 0.0 and 1.0.
    """
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())
    c = a.intersection(b)
    return (
        float(len(c)) / (len(a) + len(b) - len(c))
        if (len(a) + len(b) - len(c)) > 0
        else 0.0
    )


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    Useful for tracking metrics like loss and Jaccard score during training loops.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all internal statistics to zero."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the meter with a new value.

        Args:
            val (float): The value to add.
            n (int): The number of samples this value represents (default: 1).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
