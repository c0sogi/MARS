import nltk
import torch
import numpy as np


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking loss and metrics during training.
    """

    def __init__(self, name=None, fmt=":f"):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def compute_levenshtein(predictions, targets):
    """
    Computes the mean Levenshtein distance between a list of predicted strings
    and a list of target strings.

    Args:
        predictions (list[str]): List of predicted InChI strings.
        targets (list[str]): List of ground truth InChI strings.

    Returns:
        float: The mean Levenshtein distance.
    """
    if not predictions or not targets:
        return 0.0

    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length.")

    total_distance = 0
    for p, t in zip(predictions, targets):
        # nltk.edit_distance computes the Levenshtein distance
        total_distance += nltk.edit_distance(p, t)

    return total_distance / len(predictions)
