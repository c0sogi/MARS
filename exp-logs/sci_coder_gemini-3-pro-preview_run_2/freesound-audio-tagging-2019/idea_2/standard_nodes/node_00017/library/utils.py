import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False  # Can be set to False for absolute determinism at cost of speed
    os.environ["PYTHONHASHSEED"] = str(seed)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking losses and metrics during training.
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


def calculate_lwlrap(y_true, y_score):
    """
    Calculate the Label-Weighted Label-Ranking Average Precision (LWLRAP).

    This metric calculates the average precision for each label, and then
    averages these scores across all labels (giving each label equal weight).
    This is the primary metric for the task.

    Args:
        y_true: (n_samples, n_classes) binary ground truth (0 or 1).
        y_score: (n_samples, n_classes) predicted probabilities or logits.

    Returns:
        float: The LWLRAP score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_score, torch.Tensor):
        y_score = y_score.detach().cpu().numpy()

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    assert y_true.shape == y_score.shape, "Shapes of truth and score must match."

    num_samples, num_classes = y_true.shape

    # Sort scores in descending order (get indices)
    # axis=1 sorts along the class dimension for each sample
    # We use negative scores because argsort sorts in ascending order
    sorted_indices = np.argsort(-y_score, axis=1)

    # Create a row index grid to pair with sorted_indices
    rows = np.arange(num_samples)[:, np.newaxis]

    # Reorder truth to match the sorted score order
    # y_true_sorted[i, j] is the ground truth label of the j-th ranked class for sample i
    y_true_sorted = y_true[rows, sorted_indices]

    # Calculate cumulative sum of true labels in the sorted list
    # cumulative_true[i, j] = number of true labels in the top (j+1) predictions for sample i
    # This represents the numerator for precision at rank j+1
    cumulative_true = np.cumsum(y_true_sorted, axis=1)

    # Ranks are simply 1, 2, ..., num_classes (the denominator for precision)
    ranks = np.arange(1, num_classes + 1)

    # Precision at rank k = (number of true items in top k) / k
    precisions = cumulative_true / ranks

    # We only care about precision at the ranks where the item is actually a true label
    relevant_precisions = precisions * y_true_sorted

    # Now we map these precisions back to their original class indices.
    # We initialize an array of zeros and scatter the relevant precisions back.
    unsorted_precisions = np.zeros_like(relevant_precisions)
    unsorted_precisions[rows, sorted_indices] = relevant_precisions

    # Sum precisions for each class across all samples
    per_class_sum = np.sum(unsorted_precisions, axis=0)

    # Count number of samples containing each class (support)
    per_class_count = np.sum(y_true, axis=0)

    # Calculate average precision for each class
    # Handle division by zero for classes that don't appear in the batch/dataset
    per_class_lwlrap = np.zeros(num_classes)
    mask = per_class_count > 0
    per_class_lwlrap[mask] = per_class_sum[mask] / per_class_count[mask]

    # Final score is the average of per-class average precisions
    # This gives equal weight to each label ("Label-Weighted")
    score = np.mean(per_class_lwlrap)

    return score
