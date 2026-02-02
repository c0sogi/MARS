import os
import random
import numpy as np
import torch
from library.config import set_seed, jaccard


def seed_everything(seed=42):
    """
    Seeds all random number generators to ensure reproducibility.
    Wraps the set_seed function from the configuration library.

    Args:
        seed (int): The random seed to use.
    """
    set_seed(seed)


def normalize_text(text):
    """
    Implements the 'Normalize-First' strategy by collapsing multiple whitespaces.
    This ensures consistency between raw text and tokenization offsets.

    Args:
        text (str): The input text string.

    Returns:
        str: The normalized text with single spaces.
    """
    return " ".join(str(text).split())


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking metrics like loss and Jaccard score during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets the meter to initial state.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the meter with a new value.

        Args:
            val (float): The value to add.
            n (int): The weight of the value (usually batch size).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
