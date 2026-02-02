import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

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


def calculate_lwlrap(y_true, y_score):
    """
    Calculates Label-Weighted Label-Ranking Average Precision (lwlrap).

    This metric computes the average precision of each label (class) based on the
    ranking of the predicted scores, and then averages these per-label precisions.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth binary labels of shape (n_samples, n_classes).
        y_score (np.ndarray or torch.Tensor): Predicted probabilities of shape (n_samples, n_classes).

    Returns:
        float: The lwlrap score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_score, torch.Tensor):
        y_score = y_score.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    assert y_true.shape == y_score.shape, "y_true and y_score must have the same shape"

    n_samples, n_classes = y_true.shape

    # Sort scores in descending order.
    # argsort sorts in ascending order, so we negate y_score.
    # sorted_indices[i, :] contains the class indices sorted by score for sample i.
    sorted_indices = np.argsort(-y_score, axis=1)

    # Create row indices to index into y_true properly
    row_indices = np.arange(n_samples)[:, np.newaxis]

    # Reorder y_true based on the sorted score indices
    # y_true_sorted[i, k] is 1 if the class at rank k (0-indexed) is a positive label
    y_true_sorted = y_true[row_indices, sorted_indices]

    # Calculate cumulative sum of relevant items (positives) along the rank axis
    # cumulative_relevant[i, k] = number of positive labels in the top k+1 predictions
    cumulative_relevant = np.cumsum(y_true_sorted, axis=1)

    # Ranks are 1, 2, ..., n_classes
    ranks = np.arange(1, n_classes + 1)

    # Precision at rank k = (relevant items in top k) / k
    # precisions[i, k] is the precision at rank k+1 for sample i
    precisions = cumulative_relevant / ranks

    # We only care about the precision at ranks where the item is actually relevant
    # Filter precisions by multiplying with y_true_sorted (zeros out non-relevant ranks)
    relevant_precisions = precisions * y_true_sorted

    # Now we need to sum these precisions for each class across all samples.
    # We use sorted_indices to map the ranks back to the original class indices.

    # Flatten for efficient accumulation
    sorted_indices_flat = sorted_indices.flatten()
    relevant_precisions_flat = relevant_precisions.flatten()

    # Accumulate precisions per class
    per_class_precision_sums = np.zeros(n_classes)
    np.add.at(per_class_precision_sums, sorted_indices_flat, relevant_precisions_flat)

    # Calculate the support (number of positive samples) for each class
    per_class_support = y_true.sum(axis=0)

    # Calculate Average Precision (AP) per class
    # Handle classes with zero support to avoid division by zero
    per_class_ap = np.zeros(n_classes)
    has_support = per_class_support > 0
    per_class_ap[has_support] = (
        per_class_precision_sums[has_support] / per_class_support[has_support]
    )

    # The final score is the mean of the per-class APs (Label-Weighted means equal weight per label)
    # We average only over classes that appear in the ground truth (support > 0)
    if has_support.sum() == 0:
        return 0.0

    score = np.mean(per_class_ap[has_support])
    return score


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
