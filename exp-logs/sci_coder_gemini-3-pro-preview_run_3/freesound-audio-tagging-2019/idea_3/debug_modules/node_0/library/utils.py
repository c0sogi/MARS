import os
import copy
import torch
import numpy as np
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across numpy, torch, and python random.
    Delegates to the Config class implementation.
    """
    Config.set_seed(seed)


def calculate_lwlrap(y_true, y_score):
    """
    Calculate the Label-Weighted Label-Ranking Average Precision (LWLRAP).

    This metric calculates the average precision of retrieving the relevant labels for each class,
    and then averages these scores across all classes with equal weight.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth binary labels, shape (n_samples, n_classes).
        y_score (np.ndarray or torch.Tensor): Predicted probabilities/scores, shape (n_samples, n_classes).

    Returns:
        float: The overall LWLRAP score.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_score, torch.Tensor):
        y_score = y_score.detach().cpu().numpy()

    assert y_true.shape == y_score.shape

    num_samples, num_classes = y_score.shape

    # Sort predictions in descending order (highest score first)
    # argsort gives indices in ascending order, so we reverse the columns
    indices = np.argsort(y_score, axis=1)[:, ::-1]

    # Reorder y_true and y_score according to sorted indices
    # y_true_sorted[i, j] is the ground truth label of the j-th ranked class for sample i
    y_true_sorted = np.take_along_axis(y_true, indices, axis=1)

    # Rank is 1-based index (1, 2, ..., n_classes)
    ranks = np.arange(1, num_classes + 1)

    # Calculate precision at each rank k: (number of true labels in top k) / k
    # cumsum counts how many true labels are in the top k
    precisions = np.cumsum(y_true_sorted, axis=1) / ranks

    # We only care about precisions at ranks where the label is actually true
    relevant_precisions = precisions * y_true_sorted

    # Map relevant precisions back to their original class indices
    # We need inverse indices to map sorted positions back to original positions
    inv_indices = np.argsort(indices, axis=1)
    relevant_precisions_orig = np.take_along_axis(
        relevant_precisions, inv_indices, axis=1
    )

    # Sum the precisions for each class across all samples
    per_class_sum = np.sum(relevant_precisions_orig, axis=0)

    # Count number of samples for each class (support)
    per_class_count = np.sum(y_true, axis=0)

    # Calculate LWLRAP for each class
    # Handle division by zero for classes with no samples (safe division)
    per_class_lwlrap = np.zeros_like(per_class_sum)
    mask = per_class_count > 0
    per_class_lwlrap[mask] = per_class_sum[mask] / per_class_count[mask]

    # Average over all classes (equal weight per label)
    return np.mean(per_class_lwlrap)


def save_checkpoint(model, path):
    """
    Saves the model state dict to the specified path.
    Uses copy.deepcopy to ensure the saved state is isolated from the training loop.

    Args:
        model (torch.nn.Module): The model to save.
        path (str): The file path to save the checkpoint.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Create a deep copy of the state dict to ensure isolation
    state_dict = copy.deepcopy(model.state_dict())

    # Save to disk
    torch.save(state_dict, path)


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
