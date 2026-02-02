import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Ensures deterministic behavior for CuDNN.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_text(text):
    """
    Implements the 'Normalize-First' protocol by collapsing multiple whitespaces
    into a single space. This ensures consistency between the raw text used for
    inference and the text processed by the tokenizer.

    Args:
        text (str): The input text string.

    Returns:
        str: The normalized text with collapsed whitespace.
    """
    if not isinstance(text, str):
        text = str(text)
    return " ".join(text.split())


def jaccard(str1, str2):
    """
    Calculates the word-level Jaccard similarity score between two strings.

    Args:
        str1 (str): The first string (e.g., prediction).
        str2 (str): The second string (e.g., ground truth).

    Returns:
        float: The Jaccard score (intersection over union).
    """
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())
    c = a.intersection(b)
    return (
        float(len(c)) / (len(a) + len(b) - len(c))
        if (len(a) + len(b) - len(c)) > 0
        else 0.0
    )


class AverageMeter:
    """
    Computes and stores the average and current value of a metric.
    Used for tracking loss and scores during training and validation.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all internal statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the meter with a new value.

        Args:
            val (float): The current value to add.
            n (int): The weight or number of items associated with this value.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
