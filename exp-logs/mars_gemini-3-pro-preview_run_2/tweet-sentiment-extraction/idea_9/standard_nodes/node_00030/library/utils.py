import os
import random
import numpy as np
import torch


def seed_everything(seed):
    """
    Sets the seed for generating random numbers to ensure reproducibility
    across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def jaccard(str1, str2):
    """
    Calculates the Jaccard similarity score between two strings.

    Args:
        str1 (str): The first string (e.g., predicted text).
        str2 (str): The second string (e.g., ground truth text).

    Returns:
        float: The Jaccard similarity score (0.0 to 1.0).
    """
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())
    c = a.intersection(b)

    # Calculate Intersection over Union
    union_len = len(a) + len(b) - len(c)
    if union_len == 0:
        return 0.0
    return float(len(c)) / union_len


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training epochs.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets all internal statistics to zero.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the moving average with a new value.

        Args:
            val (float): The new value to add.
            n (int): The weight/count of the value (default: 1).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
