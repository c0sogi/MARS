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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_per_class_lwlrap(truth, scores):
    """
    Calculate label-weighted label-ranking average precision (LWLRAP) per class.

    This function computes the average precision of the ranked predictions for each class,
    averaged over the samples where that class is present in the ground truth.

    Args:
        truth (np.array or torch.Tensor): Binary ground truth matrix of shape (N_samples, N_classes).
        scores (np.array or torch.Tensor): Predicted probability matrix of shape (N_samples, N_classes).

    Returns:
        tuple: (per_class_lwlrap, weight_per_class)
            - per_class_lwlrap (np.array): LWLRAP score for each class.
            - weight_per_class (np.array): Number of samples containing each class (support).
    """
    # Convert tensors to numpy if necessary
    if torch.is_tensor(truth):
        truth = truth.cpu().numpy()
    if torch.is_tensor(scores):
        scores = scores.cpu().numpy()

    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape

    # Sort scores in descending order.
    # argsort gives ascending, so we negate scores.
    # score_indices[i, k] is the index of the class at rank k in sample i.
    score_indices = np.argsort(-scores, axis=1)

    # Reorder truth to match the ranked predictions
    # truth_sorted[i, k] is 1 if the class at rank k in sample i is a true label
    truth_sorted = np.take_along_axis(truth, score_indices, axis=1)

    # Calculate cumulative number of true labels found up to each rank
    # cumsum_truth[i, k] = number of true labels in the top (k+1) predictions
    cumsum_truth = np.cumsum(truth_sorted, axis=1)

    # Ranks are 1-based indices (1, 2, ..., num_classes)
    ranks = np.arange(1, num_classes + 1)

    # Precision at rank k = (number of true labels in top k) / k
    precisions = cumsum_truth / ranks

    # We only sum precisions for ranks that actually contain a true label
    # This matches the definition: sum_{k \in truth} Precision(k)
    valid_precisions = precisions * truth_sorted

    # Now aggregate these precisions back to their original class indices
    per_class_score = np.zeros(num_classes)

    # Flatten arrays for efficient vectorized accumulation
    flat_indices = score_indices.flatten()
    flat_precisions = valid_precisions.flatten()
    flat_truth = truth_sorted.flatten()

    # Only consider positions where there was a true label
    mask = flat_truth > 0

    # Add the precision value to the corresponding class ID
    np.add.at(per_class_score, flat_indices[mask], flat_precisions[mask])

    # Calculate the count (support) for each class
    truth_counts = truth.sum(axis=0)

    # Compute average precision per class
    # Handle division by zero for classes not present in the batch
    per_class_lwlrap = np.zeros(num_classes)
    nonzero_mask = truth_counts > 0
    per_class_lwlrap[nonzero_mask] = (
        per_class_score[nonzero_mask] / truth_counts[nonzero_mask]
    )

    return per_class_lwlrap, truth_counts


def calculate_overall_lwlrap(truth, scores):
    """
    Calculate the overall label-weighted label-ranking average precision.

    The overall score is the unweighted mean of the per-class LWLRAP scores
    for all classes present in the ground truth.

    Args:
        truth (np.array or torch.Tensor): Binary ground truth matrix.
        scores (np.array or torch.Tensor): Predicted probability matrix.

    Returns:
        float: The overall LWLRAP score.
    """
    per_class_lwlrap, truth_counts = calculate_per_class_lwlrap(truth, scores)

    # Filter out classes that have no ground truth samples in this batch/set
    valid_classes = truth_counts > 0

    if valid_classes.sum() == 0:
        return 0.0

    # Average the per-class scores
    return np.mean(per_class_lwlrap[valid_classes])


def mixup_data(x, y, alpha=0.4, device=None):
    """
    Applies Mixup augmentation to a batch of data.

    Mixup creates virtual training examples by convex combinations of pairs of inputs
    and their labels:
    x_mix = lambda * x_i + (1 - lambda) * x_j
    y_mix = lambda * y_i + (1 - lambda) * y_j

    Args:
        x (torch.Tensor): Input batch (batch_size, ...).
        y (torch.Tensor): Target batch (batch_size, ...).
        alpha (float): Mixup hyperparameter for Beta distribution.
        device (torch.device, optional): Device to perform operations on.
                                         Defaults to x.device.

    Returns:
        tuple: (mixed_x, y_a, y_b, lam)
            - mixed_x: The mixed input tensor.
            - y_a: Original targets.
            - y_b: Shuffled targets.
            - lam: The mixing coefficient lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    if device is None:
        device = x.device

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the loss for mixed inputs.

    Loss = lambda * loss(pred, y_a) + (1 - lambda) * loss(pred, y_b)

    Args:
        criterion (callable): The loss function (e.g., nn.BCEWithLogitsLoss).
        pred (torch.Tensor): Model predictions.
        y_a (torch.Tensor): Original targets.
        y_b (torch.Tensor): Shuffled targets.
        lam (float): The mixing coefficient.

    Returns:
        torch.Tensor: The calculated weighted loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
