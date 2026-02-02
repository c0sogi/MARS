import os
import torch
import numpy as np
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility using the Config class.
    """
    Config.set_seed(seed)


def calculate_lrap(y_true, y_score):
    """
    Calculate the label-weighted label-ranking average precision (LWLRAP).

    This metric calculates the average precision of retrieving relevant labels for each class,
    averaged over all classes. This gives equal weight to each label, regardless of its
    frequency in the dataset.

    Args:
        y_true (np.array or torch.Tensor): Binary ground truth labels (N_samples, N_classes).
        y_score (np.array or torch.Tensor): Predicted scores/probabilities (N_samples, N_classes).

    Returns:
        float: The LWLRAP score.
    """
    # Convert tensors to numpy if necessary
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_score):
        y_score = y_score.detach().cpu().numpy()

    assert y_true.shape == y_score.shape, "Shapes of truth and scores must match."

    # Sort scores in descending order to get ranking indices
    # argsort gives indices that would sort the array. We negate y_score to sort descending.
    sorted_indices = np.argsort(-y_score, axis=1)

    # Rearrange y_true based on the sorted score indices
    # y_true_sorted[i, j] indicates if the label at rank j (0-indexed) is true
    y_true_sorted = np.take_along_axis(y_true, sorted_indices, axis=1)

    # Calculate cumulative number of true labels at each rank
    cumulative_true = np.cumsum(y_true_sorted, axis=1)

    # Calculate precision at each rank k (1-based)
    # Precision @ k = (True positives in top k) / k
    ranks = np.arange(1, y_true.shape[1] + 1)
    precision_at_rank = cumulative_true / ranks

    # We need to map these precisions back to their original class indices
    # to compute the average precision for each specific class.
    # inverse_indices tells us where each original class ended up in the sorted list.
    inverse_indices = np.argsort(sorted_indices, axis=1)

    # Retrieve the precision corresponding to the rank of each original class
    precision_per_class_sample = np.take_along_axis(
        precision_at_rank, inverse_indices, axis=1
    )

    # We only sum precisions for the classes that are actually true for the sample
    # (i.e., we only care about the precision at the rank where a relevant item was retrieved)
    precision_per_class_sample = precision_per_class_sample * y_true

    # Sum precisions for each class across all samples
    sum_precisions_per_class = np.sum(precision_per_class_sample, axis=0)

    # Count number of samples where each class is present
    count_per_class = np.sum(y_true, axis=0)

    # Calculate average precision for each class
    # Handle division by zero for classes that never appear in the batch/dataset
    lwlrap_per_class = np.zeros_like(sum_precisions_per_class)
    mask = count_per_class > 0
    lwlrap_per_class[mask] = sum_precisions_per_class[mask] / count_per_class[mask]

    # Return the mean over all classes (label-weighted)
    return np.mean(lwlrap_per_class)


def save_checkpoint(model, optimizer, scheduler, epoch, score, filepath):
    """
    Saves the model checkpoint containing model state, optimizer state, scheduler state, and metrics.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        epoch (int): Current epoch.
        score (float): Current validation score.
        filepath (str): Path to save the checkpoint.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "score": score,
    }
    torch.save(state, filepath)


def load_checkpoint(
    filepath, model, optimizer=None, scheduler=None, device=Config.DEVICE
):
    """
    Loads a model checkpoint.

    Args:
        filepath (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The scheduler to load state into.
        device (str): Device to map the checkpoint to.

    Returns:
        tuple: (epoch, score) from the checkpoint.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint.get("epoch", 0), checkpoint.get("score", 0.0)
