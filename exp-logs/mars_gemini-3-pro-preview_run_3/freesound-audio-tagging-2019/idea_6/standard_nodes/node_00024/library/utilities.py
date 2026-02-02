import os
import random
import numpy as np
import torch
from library.configuration import Config


def set_seed(seed=42):
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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixup_data(x, y, alpha=0.4, device="cuda"):
    """
    Applies Mixup augmentation to the input batch.

    Args:
        x (torch.Tensor): Input batch of images/spectrograms.
        y (torch.Tensor): Input batch of labels.
        alpha (float): Mixup interpolation coefficient parameter.
        device (str): Device to perform calculations on.

    Returns:
        mixed_x (torch.Tensor): Mixed input batch.
        y_a (torch.Tensor): Labels for the first component.
        y_b (torch.Tensor): Labels for the second component.
        lam (float): The interpolation factor lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the Mixup loss.

    Args:
        criterion (callable): The loss function (e.g., BCEWithLogitsLoss).
        pred (torch.Tensor): Model predictions.
        y_a (torch.Tensor): Labels for the first component.
        y_b (torch.Tensor): Labels for the second component.
        lam (float): The interpolation factor lambda.

    Returns:
        torch.Tensor: The calculated loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def calculate_per_class_lwlrap(y_true, y_score):
    """
    Calculates the Label-Weighted Label-Ranking Average Precision (LWLRAP) per class.

    This metric calculates the average precision of retrieving a relevant label
    for each class, then averages these scores. It is distinct from standard LRAP
    which averages over samples.

    Args:
        y_true (np.array): Binary ground truth matrix of shape (n_samples, n_classes).
        y_score (np.array): Predicted probabilities/scores of shape (n_samples, n_classes).

    Returns:
        score (float): The overall LWLRAP score.
        weight_per_class (np.array): The LWLRAP score for each individual class.
    """
    assert y_true.shape == y_score.shape
    num_samples, num_classes = y_true.shape

    # Sort scores in descending order (highest score first)
    # argsort gives indices that would sort the array, we want descending
    score_indices = np.argsort(-y_score, axis=1)

    # Create an array of ranks (0, 1, 2, ...)
    # shape: (num_samples, num_classes)
    ranks = np.arange(num_classes)[None, :]

    # Reorder y_true according to the sorted score indices
    # This tells us which labels are present at each rank position
    y_true_sorted = y_true[np.arange(num_samples)[:, None], score_indices]

    # Calculate cumulative sum of true positives along the ranked list
    # At index i, this is the number of relevant items found in ranks 0...i
    cumulative_tp = np.cumsum(y_true_sorted, axis=1)

    # Calculate precision at each rank k: P@k = (relevant items in top k) / k
    # Ranks are 0-indexed, so we add 1 for division
    precisions = cumulative_tp / (ranks + 1)

    # We only care about the precisions at the ranks where the label is actually positive
    # If y_true_sorted is 0, that rank doesn't contribute to the Average Precision for that sample's labels
    relevant_precisions_sorted = precisions * y_true_sorted

    # Now we need to map these precisions back to their original class indices to sum them per class
    # We create a placeholder array
    relevant_precisions = np.zeros_like(y_score)

    # Scatter the sorted precisions back to their original positions
    # y_score indices: [row_idx, original_class_idx]
    # We use the score_indices we computed earlier
    relevant_precisions[np.arange(num_samples)[:, None], score_indices] = (
        relevant_precisions_sorted
    )

    # Sum precisions for each class across all samples
    per_class_sum = relevant_precisions.sum(axis=0)

    # Count how many times each class appears in the ground truth
    per_class_count = y_true.sum(axis=0)

    # Avoid division by zero for classes that don't appear in the batch/dataset
    # If a class has 0 samples, its score is 0
    per_class_score = per_class_sum / np.maximum(per_class_count, 1)

    # Overall score is the unweighted average of per-class scores
    score = per_class_score.mean()

    return score, per_class_score


def calculate_lrap(y_true, y_score):
    """
    Wrapper function to calculate the scalar LWLRAP score.

    Args:
        y_true (np.array): Binary ground truth matrix.
        y_score (np.array): Predicted probabilities.

    Returns:
        float: The LWLRAP score.
    """
    score, _ = calculate_per_class_lwlrap(y_true, y_score)
    return score
