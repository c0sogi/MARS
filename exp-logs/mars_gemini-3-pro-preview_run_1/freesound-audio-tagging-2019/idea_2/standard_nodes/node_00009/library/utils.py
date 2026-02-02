import os
import random
import numpy as np
import torch
from library.config import CFG


def set_seed(seed=CFG.seed):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_per_class_lwlrap(truth, scores):
    """
    Calculates the Label-Weighted Label-Ranking Average Precision (LWLRAP) for each class.

    This metric computes the average precision of the ranked predictions for each label,
    averaged over the samples where that label is present.

    Args:
        truth (np.array or torch.Tensor): Binary ground truth labels of shape (N_samples, N_classes).
        scores (np.array or torch.Tensor): Predicted probabilities of shape (N_samples, N_classes).

    Returns:
        score_per_class (np.array): The LWLRAP score for each class.
        weight_per_class (np.array): The number of positive samples (support) for each class.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(truth, torch.Tensor):
        truth = truth.detach().cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()

    assert truth.shape == scores.shape, "Truth and scores must have the same shape"
    num_samples, num_classes = scores.shape

    # Sort scores in descending order
    # indices[i, k] gives the column index of the k-th highest score for sample i
    indices = np.argsort(-scores, axis=1)

    # Reorder truth labels according to the sorted scores
    # truth_sorted[i, k] is the ground truth label of the class ranked k-th in sample i
    truth_sorted = np.take_along_axis(truth, indices, axis=1)

    # Calculate the cumulative number of true positives encountered up to each rank
    cumulative_hits = np.cumsum(truth_sorted, axis=1)

    # Calculate precision at each rank (1-based ranking)
    # precisions[i, k] is the precision at rank k+1 for sample i
    precisions = cumulative_hits / np.arange(1, num_classes + 1)

    # Map the calculated precisions back to their original class indices
    # precisions_mapped[i, c] will store the precision at the rank where class c was placed
    precisions_mapped = np.zeros_like(scores)
    np.put_along_axis(precisions_mapped, indices, precisions, axis=1)

    # We only care about the precision values for classes that are actually positive (truth == 1)
    relevant_precisions = precisions_mapped * truth

    # Sum the precisions for each class across all samples
    sum_precisions_per_class = relevant_precisions.sum(axis=0)

    # Count the number of positive samples for each class
    weight_per_class = truth.sum(axis=0)

    # Calculate the average precision per class
    # Handle classes with no positive samples to avoid division by zero
    score_per_class = np.zeros(num_classes)
    mask = weight_per_class > 0
    score_per_class[mask] = sum_precisions_per_class[mask] / weight_per_class[mask]

    return score_per_class, weight_per_class


def calculate_overall_lwlrap(truth, scores):
    """
    Calculates the overall Label-Weighted Label-Ranking Average Precision.

    The overall score is the unweighted mean of the per-class LWLRAP scores.

    Args:
        truth (np.array or torch.Tensor): Binary ground truth labels.
        scores (np.array or torch.Tensor): Predicted probabilities.

    Returns:
        float: The overall LWLRAP score.
    """
    score_per_class, _ = calculate_per_class_lwlrap(truth, scores)

    # Calculate the average over all classes (Label-Weighted)
    return np.mean(score_per_class)
