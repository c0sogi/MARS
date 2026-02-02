import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Sets the seed for generating random numbers to ensure reproducibility
    across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def jaccard(str1: str, str2: str) -> float:
    """
    Calculates the Jaccard similarity score between two strings based on word overlap.

    Args:
        str1 (str): The first string (e.g., prediction).
        str2 (str): The second string (e.g., ground truth).

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


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
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

    def update(self, val: float, n: int = 1):
        """
        Updates the moving average with a new value.

        Args:
            val (float): The new value to add.
            n (int): The weight/count of the value (default is 1).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
