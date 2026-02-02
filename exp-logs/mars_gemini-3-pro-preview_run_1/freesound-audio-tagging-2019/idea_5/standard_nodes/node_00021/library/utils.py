import os
import sys
import random
import numpy as np
import torch
import logging
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name):
    """
    Creates and returns a logger with standard formatting.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if function is called repeatedly
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
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


def calculate_per_class_lwlrap(truth, scores):
    """
    Calculate label-weighted label-ranking average precision per class.

    Arguments:
        truth: np.array or torch.Tensor of (num_clips, num_classes) containing 0 or 1.
        scores: np.array or torch.Tensor of (num_clips, num_classes) containing float probabilities.

    Returns:
        per_class_lwlrap: np.array of (num_classes,) giving the lwlrap for each class.
        weight_per_class: np.array of (num_classes,) giving the count of each class.
    """
    # Convert tensors to numpy if necessary
    if isinstance(truth, torch.Tensor):
        truth = truth.detach().cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()

    assert (
        truth.shape == scores.shape
    ), f"Shape mismatch: truth {truth.shape} vs scores {scores.shape}"
    num_samples, num_classes = scores.shape

    # Sort scores descending (argsort gives indices of sorted elements)
    # We negate scores to sort descending
    score_indices = np.argsort(-scores, axis=1)

    # Reorder truth to match the sorted scores
    # np.take_along_axis is efficient for this specific gathering operation
    truth_permuted = np.take_along_axis(truth, score_indices, axis=1)

    # Cumulative sum of true labels along the sorted axis
    # This tells us, for each rank k, how many true labels are in the top k
    cum_true = np.cumsum(truth_permuted, axis=1)

    # Ranks are 1, 2, ..., C
    ranks = np.arange(1, num_classes + 1)

    # Precision at each rank k is (number of true labels in top k) / k
    precisions = cum_true / ranks[None, :]

    # We only care about the precision at the ranks where the label is actually True
    relevant_precisions = precisions * truth_permuted

    # Now we map these relevant precisions back to their original class indices to sum them up.
    # score_indices tells us which class is at which rank.

    flat_indices = score_indices.flatten()
    flat_precisions = relevant_precisions.flatten()

    # Sum precisions for each class
    class_precisions_sum = np.bincount(
        flat_indices, weights=flat_precisions, minlength=num_classes
    )

    # Count number of positive samples for each class
    class_counts = truth.sum(axis=0)

    # Calculate LWLRAP per class (avoid division by zero)
    per_class_lwlrap = np.zeros(num_classes)
    mask = class_counts > 0
    per_class_lwlrap[mask] = class_precisions_sum[mask] / class_counts[mask]

    return per_class_lwlrap, class_counts


def calculate_lwlrap(truth, scores):
    """
    Calculate the overall Label-Weighted Label-Ranking Average Precision.
    This is the primary metric for the task.

    Arguments:
        truth: np.array or torch.Tensor of (num_clips, num_classes).
        scores: np.array or torch.Tensor of (num_clips, num_classes).

    Returns:
        float: The mean LWLRAP across all classes.
    """
    per_class_lwlrap, _ = calculate_per_class_lwlrap(truth, scores)
    # The metric is the unweighted mean of the per-class LWLRAP scores
    return np.mean(per_class_lwlrap)
