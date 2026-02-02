import os
import random
import numpy as np
import torch
from nltk.metrics import edit_distance


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and accuracy during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def seed_everything(seed: int = 42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_levenshtein(predictions: list, references: list) -> float:
    """
    Computes the mean Levenshtein distance between predictions and references.

    Args:
        predictions (list[str]): List of predicted strings.
        references (list[str]): List of ground truth strings.

    Returns:
        float: The mean Levenshtein distance.
    """
    if not predictions:
        return 0.0

    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: predictions({len(predictions)}) != references({len(references)})"
        )

    total_dist = 0
    for p, r in zip(predictions, references):
        total_dist += edit_distance(p, r)

    return total_dist / len(predictions)
