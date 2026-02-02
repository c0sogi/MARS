import nltk
from typing import List, Union


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking loss and other metrics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_levenshtein(preds: List[str], targets: List[str]) -> float:
    """
    Computes the mean Levenshtein distance between predictions and targets.

    Args:
        preds (List[str]): List of predicted InChI strings.
        targets (List[str]): List of ground truth InChI strings.

    Returns:
        float: The mean Levenshtein distance.
    """
    if not preds:
        return 0.0

    # Ensure lists are of the same length
    if len(preds) != len(targets):
        raise ValueError(
            f"Predictions ({len(preds)}) and targets ({len(targets)}) must have the same length."
        )

    total_distance = 0
    count = len(preds)

    for p, t in zip(preds, targets):
        # nltk.edit_distance computes the Levenshtein distance
        dist = nltk.edit_distance(p, t)
        total_distance += dist

    return total_distance / count
