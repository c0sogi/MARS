import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

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


def calculate_per_class_lwlrap(y_true, y_score):
    """
    Computes Label-Weighted Label-Ranking Average Precision (LWLRAP) per class.

    This function calculates the average precision of the ranked predictions for each class,
    averaged over the samples where that class is present.

    Args:
        y_true (np.array or torch.Tensor): Binary ground truth matrix (n_samples, n_classes).
        y_score (np.array or torch.Tensor): Predicted probabilities (n_samples, n_classes).

    Returns:
        per_class_lwlrap (np.array): The LWLRAP score for each class.
        per_class_counts (np.array): The number of positive samples for each class.
    """
    # Convert tensors to numpy if necessary
    if hasattr(y_true, "cpu"):
        y_true = y_true.detach().cpu().numpy()
    if hasattr(y_score, "cpu"):
        y_score = y_score.detach().cpu().numpy()

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    n_samples, n_classes = y_true.shape

    # Sort scores descending to get rankings
    # argsort gives indices that sort the array. We sort -y_score to get descending order.
    sorted_indices = np.argsort(-y_score, axis=1)

    # Create row indices for advanced indexing
    row_indices = np.arange(n_samples)[:, np.newaxis]

    # Reorder y_true based on the sorted scores
    # This matrix tells us if the label at rank k is a true label
    y_true_sorted = y_true[row_indices, sorted_indices]

    # Compute cumulative sum of true positives along the ranked list
    # cumulative_tp[i, k] = number of true labels in the top k+1 predictions for sample i
    cumulative_tp = np.cumsum(y_true_sorted, axis=1)

    # Compute ranks (1-based)
    ranks = np.arange(1, n_classes + 1)

    # Calculate precision at each rank
    # precisions[i, k] = precision at rank k+1 for sample i
    precisions = cumulative_tp / ranks

    # We only care about the precisions at the ranks where the label is actually True
    # Mask out precisions for false labels
    relevant_precisions = precisions * y_true_sorted

    # Map these precisions back to their original class indices
    # We want to accumulate the precision achieved for specific classes
    mapped_precisions = np.zeros_like(relevant_precisions)

    # Scatter the values back: mapped_precisions[i, original_class_idx] = precision
    mapped_precisions[row_indices, sorted_indices] = relevant_precisions

    # Sum the precisions for each class across all samples
    per_class_precision_sums = np.sum(mapped_precisions, axis=0)

    # Count the number of samples for each class
    per_class_counts = np.sum(y_true, axis=0)

    # Avoid division by zero
    safe_counts = np.maximum(per_class_counts, 1)

    # Calculate average precision per class
    per_class_lwlrap = per_class_precision_sums / safe_counts

    return per_class_lwlrap, per_class_counts


def calculate_lwlrap(y_true, y_score):
    """
    Computes the overall Label-Weighted Label-Ranking Average Precision.
    The overall score is the unweighted average of the per-class LWLRAP scores.

    Args:
        y_true (np.array or torch.Tensor): Binary ground truth matrix (n_samples, n_classes).
        y_score (np.array or torch.Tensor): Predicted probabilities (n_samples, n_classes).

    Returns:
        float: The overall LWLRAP score.
    """
    per_class_lwlrap, _ = calculate_per_class_lwlrap(y_true, y_score)
    return np.mean(per_class_lwlrap)
