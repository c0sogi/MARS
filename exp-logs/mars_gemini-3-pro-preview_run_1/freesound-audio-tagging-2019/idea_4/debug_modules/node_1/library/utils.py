import os
import random
import numpy as np
import torch
from library.config import Config


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


def calculate_per_class_lwlrap(truth, scores):
    """
    Calculates the Label-Weighted Label-Ranking Average Precision (LWLRAP).

    This metric computes the average precision of the ranked retrieval list for each
    label, averages it for that specific label, and then computes the unweighted mean
    across all labels.

    Args:
        truth (np.array or torch.Tensor): Binary ground truth matrix (num_samples, num_classes).
        scores (np.array or torch.Tensor): Predicted probabilities (num_samples, num_classes).

    Returns:
        tuple: (overall_lwlrap, per_class_lwlrap_array)
    """
    # Convert tensors to numpy if necessary
    if isinstance(truth, torch.Tensor):
        truth = truth.detach().cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()

    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape

    # Sort scores in descending order to determine ranks
    # argsort returns indices that sort the array. We negate scores for descending sort.
    sorted_indices = np.argsort(-scores, axis=1)

    # Reorder truth matrix according to the sorted scores
    # sorted_truth[i, j] is 1 if the label at rank j (0-based) for sample i is a true label
    sorted_truth = np.take_along_axis(truth, sorted_indices, axis=1)

    # Calculate cumulative true positives at each rank
    # cumulative_tp[i, j] = number of true labels in the top (j+1) predictions for sample i
    cumulative_tp = np.cumsum(sorted_truth, axis=1)

    # Ranks are simply 1-based indices (1, 2, ..., num_classes)
    ranks = np.arange(1, num_classes + 1)

    # Precision at rank k = (TP in top k) / k
    precisions = cumulative_tp / ranks

    # We only sum precisions at ranks where the label is actually positive (relevant items)
    relevant_precisions = precisions * sorted_truth

    # Now we need to aggregate these precisions back to their original class indices.
    # sorted_indices maps (sample, rank) -> class_id.
    # We flatten the arrays to use np.add.at for fast aggregation.
    flat_indices = sorted_indices.flatten()
    flat_precisions = relevant_precisions.flatten()

    # Accumulate precision sums for each class
    class_prec_sums = np.zeros(num_classes, dtype=np.float64)
    np.add.at(class_prec_sums, flat_indices, flat_precisions)

    # Calculate the total number of positive samples for each class
    class_counts = truth.sum(axis=0)

    # Avoid division by zero for classes that don't appear in the batch/dataset
    # If count is 0, the sum is 0, so result is 0.
    safe_class_counts = np.maximum(class_counts, 1)

    # Average precision for each class
    per_class_lwlrap = class_prec_sums / safe_class_counts

    # Overall LWLRAP is the mean over all classes (equal weight per label)
    # Note: We average over all classes, even those not present (score 0),
    # though typically in validation all classes should be present.
    overall_lwlrap = np.mean(per_class_lwlrap)

    return overall_lwlrap, per_class_lwlrap


def calculate_lwlrap(y_true, y_score):
    """
    Wrapper function to calculate the scalar LWLRAP score.

    Args:
        y_true (np.array or torch.Tensor): Binary ground truth matrix.
        y_score (np.array or torch.Tensor): Predicted probabilities.

    Returns:
        float: The label-weighted label-ranking average precision.
    """
    score, _ = calculate_per_class_lwlrap(y_true, y_score)
    return score


def save_checkpoint(
    model, optimizer, scheduler, epoch, score, filename="best_model.pth"
):
    """
    Saves the model checkpoint to the configured output directory.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler state.
        epoch (int): Current epoch number.
        score (float): Validation score (LWLRAP).
        filename (str): Name of the file.
    """
    save_path = os.path.join(Config.OUTPUT_DIR, filename)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "score": score,
    }
    torch.save(state, save_path)


def load_checkpoint(model, optimizer=None, scheduler=None, filename="best_model.pth"):
    """
    Loads a model checkpoint from the configured output directory.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler to load state into.
        filename (str): Name of the file to load.

    Returns:
        float: The score recorded in the checkpoint, or 0.0 if not found.
    """
    load_path = os.path.join(Config.OUTPUT_DIR, filename)
    if not os.path.exists(load_path):
        return 0.0

    checkpoint = torch.load(load_path, map_location=Config.DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if (
        scheduler
        and "scheduler_state_dict" in checkpoint
        and checkpoint["scheduler_state_dict"]
    ):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint.get("score", 0.0)
