import os
import random
import numpy as np
import torch


def seed_everything(seed):
    """
    Seeds all random number generators to ensure reproducibility.

    Args:
        seed (int): The random seed to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixup_data(x, y, alpha=1.0):
    """
    Applies Mixup augmentation to the batch.

    Args:
        x (torch.Tensor): Input batch (e.g., spectrograms).
        y (torch.Tensor): Target batch (labels).
        alpha (float): Mixup alpha parameter.

    Returns:
        mixed_x (torch.Tensor): The mixed input data.
        y_a (torch.Tensor): The original targets.
        y_b (torch.Tensor): The shuffled targets.
        lam (float): The mixing coefficient.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    # Generate random permutation on the same device as input
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the loss for Mixup.

    Args:
        criterion (callable): The loss function (e.g., BCEWithLogitsLoss).
        pred (torch.Tensor): Model predictions.
        y_a (torch.Tensor): Original targets.
        y_b (torch.Tensor): Shuffled targets.
        lam (float): Mixing coefficient.

    Returns:
        torch.Tensor: The weighted loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def calculate_per_class_lwlrap(truth, scores):
    """
    Calculates the label-weighted label-ranking average precision (LWLRAP) per class.

    Args:
        truth (np.array or torch.Tensor): Binary ground truth matrix of shape (n_samples, n_classes).
        scores (np.array or torch.Tensor): Predicted probability matrix of shape (n_samples, n_classes).

    Returns:
        per_class_score (np.array): The LWLRAP score for each class.
    """
    # Ensure inputs are numpy arrays
    if torch.is_tensor(truth):
        truth = truth.cpu().numpy()
    if torch.is_tensor(scores):
        scores = scores.cpu().numpy()

    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape

    # Sort scores in descending order (argsort gives ascending, so reverse)
    score_ranks = np.argsort(scores, axis=1)[:, ::-1]

    # Reorder truth to match the sorted scores
    truth_sorted = np.take_along_axis(truth, score_ranks, axis=1)

    # Calculate cumulative number of relevant items at each rank
    cumulative_truth = np.cumsum(truth_sorted, axis=1)

    # Calculate precision at each rank: (relevant items found) / (rank)
    # Ranks are 1-based
    ranks = np.arange(1, num_classes + 1)
    precisions = cumulative_truth / ranks

    # We only care about precisions at ranks where the item is actually relevant
    relevant_precisions = precisions * truth_sorted

    # Map these precisions back to their original class indices
    mapped_precisions = np.zeros_like(relevant_precisions)
    np.put_along_axis(mapped_precisions, score_ranks, relevant_precisions, axis=1)

    # Calculate per-class score: sum of precisions / count of relevant items
    class_counts = truth.sum(axis=0)
    per_class_sum = mapped_precisions.sum(axis=0)

    # Handle classes with no positive samples in this batch to avoid division by zero
    per_class_score = np.zeros(num_classes)
    mask = class_counts > 0
    per_class_score[mask] = per_class_sum[mask] / class_counts[mask]

    return per_class_score


def calculate_overall_lwlrap(truth, scores):
    """
    Calculates the overall label-weighted label-ranking average precision.

    Args:
        truth (np.array or torch.Tensor): Binary ground truth matrix.
        scores (np.array or torch.Tensor): Predicted probability matrix.

    Returns:
        float: The mean LWLRAP across all classes.
    """
    per_class_score = calculate_per_class_lwlrap(truth, scores)
    return np.mean(per_class_score)
