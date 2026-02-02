import os
import random
import numpy as np
import torch
import nltk


def seed_everything(seed: int = 42):
    """
    Seeds all random number generators to ensure reproducibility.

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


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking losses and metrics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        Update the meter with a new value.

        Args:
            val (float): The value to add.
            n (int): The number of samples this value represents (weight).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_levenshtein(predictions: list, targets: list) -> float:
    """
    Computes the mean Levenshtein distance between predictions and targets.

    Args:
        predictions (list of str): List of predicted strings.
        targets (list of str): List of ground truth strings.

    Returns:
        float: The mean Levenshtein distance.
    """
    if not predictions or not targets:
        return 0.0

    if len(predictions) != len(targets):
        raise ValueError(
            f"Size mismatch: predictions ({len(predictions)}) vs targets ({len(targets)})"
        )

    distances = []
    for pred, target in zip(predictions, targets):
        # Ensure inputs are strings
        p = str(pred)
        t = str(target)
        # nltk.edit_distance computes Levenshtein distance
        d = nltk.edit_distance(p, t)
        distances.append(d)

    return float(np.mean(distances))
