import os
import random
import numpy as np
import torch


class AverageMeter(object):
    """Computes and stores the average and current value."""

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


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_lwlrap(y_true, y_score):
    """
    Calculate the Label-Weighted Label-Ranking Average Precision (lwlrap).

    This metric is the average over all labels in the test set, where each label
    receives equal weight.

    Args:
        y_true (np.array or torch.Tensor): Binary ground truth matrix of shape (N_samples, N_classes).
        y_score (np.array or torch.Tensor): Predicted probabilities of shape (N_samples, N_classes).

    Returns:
        float: The scalar lwlrap score.
    """
    # Convert tensors to numpy if necessary
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_score):
        y_score = y_score.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    assert y_true.shape == y_score.shape, "Shapes of truth and score must match."

    num_samples, num_classes = y_score.shape

    # Sort scores descending
    # argsort gives indices that sort the array ascending, [:, ::-1] reverses to descending
    score_ranks = np.argsort(y_score, axis=1)[:, ::-1]

    # Rearrange truth according to score ranks to see if high-ranked items are true
    # row_indices: (N, 1) broadcasted against score_ranks (N, C)
    row_indices = np.arange(num_samples)[:, np.newaxis]
    sorted_truth = y_true[row_indices, score_ranks]

    # Calculate precision at each rank k (1-based)
    # cumsum counts how many true positives are in the top k
    precisions = np.cumsum(sorted_truth, axis=1) / np.arange(1, num_classes + 1)

    # We only care about precisions at ranks where the label is actually relevant (truth=1).
    # We need to map these precisions back to their original class indices to average per class.

    # Create a matrix to hold precisions aligned with original class indices
    precisions_at_original_indices = np.zeros_like(precisions)

    # Map the calculated precisions back to the original class columns
    precisions_at_original_indices[row_indices, score_ranks] = precisions

    # Filter: only keep precisions for classes that are actually positive for the sample
    relevant_precisions = precisions_at_original_indices * y_true

    # Sum precisions over all samples for each class
    per_class_sum = relevant_precisions.sum(axis=0)

    # Count number of positive samples for each class
    per_class_count = y_true.sum(axis=0)

    # Calculate average precision for each class
    # Handle classes with 0 positive samples by masking
    valid_classes = per_class_count > 0
    per_class_lwlrap = np.zeros(num_classes)

    if np.any(valid_classes):
        per_class_lwlrap[valid_classes] = (
            per_class_sum[valid_classes] / per_class_count[valid_classes]
        )
        # The final score is the average of the per-class scores
        return np.mean(per_class_lwlrap[valid_classes])
    else:
        return 0.0
