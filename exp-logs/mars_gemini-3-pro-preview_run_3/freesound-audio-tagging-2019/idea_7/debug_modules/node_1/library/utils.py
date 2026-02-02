import os
import random
import copy
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

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_lwlrap(y_true, y_score):
    """
    Calculate the Label-Weighted Label-Ranking Average Precision (LWLRAP).

    This metric calculates the average precision of predictions for each label,
    and then averages these scores across all labels, giving each label equal weight.
    This implementation uses vectorized PyTorch operations for efficiency.

    Args:
        y_true (np.ndarray or torch.Tensor): Binary ground truth matrix of shape (N_samples, N_classes).
        y_score (np.ndarray or torch.Tensor): Predicted probability matrix of shape (N_samples, N_classes).

    Returns:
        float: The LWLRAP score.
    """
    # Convert inputs to torch tensors if they are numpy arrays
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if isinstance(y_score, np.ndarray):
        y_score = torch.from_numpy(y_score)

    # Ensure inputs are float32
    y_true = y_true.float()
    y_score = y_score.float()

    # Move y_true to the same device as y_score
    device = y_score.device
    y_true = y_true.to(device)

    batch_size, num_classes = y_true.shape

    # Sort scores in descending order to determine ranking
    # indices: (N, C) containing the original class indices sorted by score
    _, indices = torch.sort(y_score, dim=1, descending=True)

    # Gather ground truth labels in the order of the predicted rankings
    # y_true_sorted[i, j] is the truth value of the class ranked j-th in sample i
    y_true_sorted = torch.gather(y_true, 1, indices)

    # Calculate cumulative true positives at each rank
    # num_hits_at_k[i, j] = number of relevant items in the top (j+1) predictions
    num_hits_at_k = y_true_sorted.cumsum(dim=1)

    # Create a tensor of ranks (1-based)
    # ranks: (1, C) -> broadcasted to (N, C)
    ranks = torch.arange(1, num_classes + 1, device=device, dtype=torch.float32).view(
        1, -1
    )

    # Calculate precision at each rank
    precisions = num_hits_at_k / ranks

    # We only care about the precision at ranks where the label is actually relevant
    # (i.e., where y_true_sorted is 1)
    relevant_precisions = precisions * y_true_sorted

    # Now we need to map these precisions back to their original class indices
    # to calculate the average precision for each class.
    # We use scatter_ to place the precision values back into their original columns.
    mapped_precisions = torch.zeros_like(y_true)
    mapped_precisions.scatter_(1, indices, relevant_precisions)

    # Sum the precisions for each class across all samples in the batch
    per_class_precision_sum = mapped_precisions.sum(dim=0)

    # Count the total number of positive samples for each class in the batch
    per_class_count = y_true.sum(dim=0)

    # Calculate the average precision for each class
    # Clamp count to 1.0 to avoid division by zero for absent classes
    per_class_lwlrap = per_class_precision_sum / per_class_count.clamp(min=1.0)

    # Identify classes that are present in the ground truth
    present_mask = per_class_count > 0

    # If no classes are present (e.g., empty batch or all zeros), return 0
    if present_mask.sum() == 0:
        return 0.0

    # Compute the macro-average over the present classes
    score = per_class_lwlrap[present_mask].mean()

    return score.item()


def save_checkpoint(model, path):
    """
    Saves a deep copy of the model's state dictionary to the specified path.

    Using deepcopy ensures that the saved weights are exactly those at the time
    of the function call, preventing issues if the model is modified in memory
    subsequently.

    Args:
        model (torch.nn.Module): The model instance to save.
        path (str): The destination file path.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Create a deep copy of the state dict
    state_dict = copy.deepcopy(model.state_dict())

    # Save to disk
    torch.save(state_dict, path)
