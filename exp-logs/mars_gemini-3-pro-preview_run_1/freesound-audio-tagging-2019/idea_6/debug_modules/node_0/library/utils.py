import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class AverageMeter:
    """
    Computes and stores the average and current value.
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


def calculate_per_class_lwlrap(y_true, y_score):
    """
    Calculate label-weighted label-ranking average precision per class.

    Args:
        y_true (array-like): Binary ground truth labels (n_samples, n_classes).
        y_score (array-like): Predicted scores/probabilities (n_samples, n_classes).

    Returns:
        np.array: lwlrap score for each class (n_classes,).
    """
    # Convert to numpy if tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_score, torch.Tensor):
        y_score = y_score.detach().cpu().numpy()

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    assert y_true.shape == y_score.shape
    n_samples, n_classes = y_true.shape

    # Sort scores descending
    # argsort gives indices that sort the array. We want descending, so -y_score.
    score_indices = np.argsort(-y_score, axis=1)

    # Reorder truth based on score ranks
    y_true_sorted = np.take_along_axis(y_true, score_indices, axis=1)

    # Ranks are 1-based
    ranks = np.arange(1, n_classes + 1)

    # Cumulative sum of positive labels along the ranked list
    # cum_pos[i, j] is the number of true positives in the top (j+1) predictions for sample i
    cum_pos = np.cumsum(y_true_sorted, axis=1)

    # Precision at each rank k is (number of true positives in top k) / k
    precisions = cum_pos / ranks[None, :]

    # We only care about precision at ranks where the label is actually positive
    relevant_precisions = precisions * y_true_sorted

    # Now we aggregate these precisions back to their original class buckets.
    # score_indices[i, j] is the class ID at rank j for sample i.
    # relevant_precisions[i, j] is the precision contribution for that class instance.

    per_class_score = np.zeros(n_classes)
    per_class_count = np.zeros(n_classes)

    # Flatten arrays for efficient accumulation
    indices_flat = score_indices.flatten()
    prec_flat = relevant_precisions.flatten()
    truth_flat = y_true_sorted.flatten()

    # Accumulate scores and counts
    np.add.at(per_class_score, indices_flat, prec_flat)
    np.add.at(per_class_count, indices_flat, truth_flat)

    # Calculate average precision per class
    # Handle division by zero for classes not present in the batch/dataset
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_lwlrap = per_class_score / per_class_count

    # Replace NaNs (classes with count 0) with 0
    per_class_lwlrap = np.nan_to_num(per_class_lwlrap)

    return per_class_lwlrap


def calculate_lwlrap(y_true, y_score):
    """
    Calculate the overall Label-Weighted Label-Ranking Average Precision.

    Args:
        y_true (array-like): Binary ground truth labels.
        y_score (array-like): Predicted scores.

    Returns:
        float: The mean lwlrap across all classes.
    """
    per_class_lwlrap = calculate_per_class_lwlrap(y_true, y_score)
    return np.mean(per_class_lwlrap)
