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
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def mixup_data(x, y, alpha=0.4, device="cuda"):
    """
    Applies Mixup augmentation to the batch.

    Args:
        x (torch.Tensor): Input batch of images/spectrograms.
        y (torch.Tensor): Input batch of targets.
        alpha (float): Mixup interpolation coefficient parameter.
        device (str): Device to perform the operation on.

    Returns:
        mixed_x (torch.Tensor): The mixed input batch.
        y_a (torch.Tensor): Targets for the first set of images.
        y_b (torch.Tensor): Targets for the second set of images (shuffled).
        lam (float): The lambda value used for mixing.
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
    Calculates the Mixup loss using the weighted sum of losses.

    Args:
        criterion (callable): The loss function.
        pred (torch.Tensor): Model predictions.
        y_a (torch.Tensor): Targets for the first set.
        y_b (torch.Tensor): Targets for the second set.
        lam (float): The lambda value used for mixing.

    Returns:
        torch.Tensor: The calculated loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def calculate_per_class_lwlrap(truth, scores):
    """
    Calculates the label-weighted label-ranking average precision (lwlrap) per class.

    Args:
        truth (np.ndarray): Binary ground truth labels (N_samples, N_classes).
        scores (np.ndarray): Predicted probabilities (N_samples, N_classes).

    Returns:
        score (float): The overall lwlrap score (average across classes).
        per_class_lwlrap (np.ndarray): The lwlrap score for each class.
    """
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape

    # Sort scores in descending order (argsort gives ascending, so we negate scores)
    sorted_indices = np.argsort(-scores, axis=1)

    # Reorder truth to match sorted scores
    truth_sorted = np.take_along_axis(truth, sorted_indices, axis=1)

    # Cumulative sum of truth along the sorted axis (numerator for precision)
    # cum_truth[i, j] tells us how many true labels are in the top j+1 predictions
    cum_truth = np.cumsum(truth_sorted, axis=1)

    # Ranks (denominator for precision)
    ranks = np.arange(1, num_classes + 1)

    # Precision at each rank
    precisions = cum_truth / ranks[None, :]

    # Filter precisions: only consider ranks where the item is actually a true label
    relevant_precisions = precisions * truth_sorted

    # Map back to original class indices to sum per class
    inverse_indices = np.argsort(sorted_indices, axis=1)
    per_class_precisions = np.take_along_axis(
        relevant_precisions, inverse_indices, axis=1
    )

    # Sum precisions per class
    per_class_sums = np.sum(per_class_precisions, axis=0)

    # Count positive samples per class
    per_class_counts = np.sum(truth, axis=0)

    # Calculate lwlrap per class (handle division by zero)
    per_class_lwlrap = per_class_sums / np.maximum(1.0, per_class_counts)

    # Overall score is the mean over classes (Label-Weighted)
    overall_lwlrap = np.mean(per_class_lwlrap)

    return overall_lwlrap, per_class_lwlrap


def calculate_lwlrap(truth, scores):
    """
    Calculates the overall label-weighted label-ranking average precision.
    Wrapper around calculate_per_class_lwlrap.

    Args:
        truth (np.ndarray): Binary ground truth labels.
        scores (np.ndarray): Predicted probabilities.

    Returns:
        float: The overall lwlrap score.
    """
    score, _ = calculate_per_class_lwlrap(truth, scores)
    return score
