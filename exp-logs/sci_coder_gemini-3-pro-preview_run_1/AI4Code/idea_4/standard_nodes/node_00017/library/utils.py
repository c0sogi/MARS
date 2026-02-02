import os
import random
import torch
import numpy as np
from bisect import bisect
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_inversions(a):
    """
    Counts the number of inversions in a list of integers.
    An inversion is a pair (i, j) such that i < j and a[i] > a[j].

    Args:
        a (list): List of integers (ranks).

    Returns:
        int: Number of inversions.
    """
    inversions = 0
    sorted_so_far = []
    for x in a:
        # Find position to insert x to maintain sorted order
        idx = bisect(sorted_so_far, x)
        # Elements after idx in sorted_so_far are greater than x but appeared earlier
        inversions += len(sorted_so_far) - idx
        sorted_so_far.insert(idx, x)
    return inversions


def compute_kendall_tau(predictions, ground_truths):
    """
    Computes the Kendall Tau correlation metric accumulated across a collection of notebooks.

    Formula: K = 1 - 4 * (Sum of Swaps) / (Sum of n*(n-1))

    Args:
        predictions (list of list): Predicted cell orders (list of cell IDs).
        ground_truths (list of list): Ground truth cell orders (list of cell IDs).

    Returns:
        float: The computed Kendall Tau score.
    """
    total_inversions = 0
    total_pairs = 0  # This corresponds to sum of n*(n-1)

    for pred, gt in zip(predictions, ground_truths):
        n = len(gt)
        if n <= 1:
            continue

        # Map cell IDs to their ground truth rank (0 to n-1)
        rank_map = {cell_id: r for r, cell_id in enumerate(gt)}

        # Convert prediction sequence to rank sequence
        # Filter to ensure we only consider cells present in the ground truth
        # (Handling potential mismatches gracefully, though inputs should be valid)
        pred_ranks = [rank_map[cid] for cid in pred if cid in rank_map]

        # If lengths mismatch significantly, it might indicate an issue,
        # but we proceed with the ranks we found.

        # Count swaps (inversions) needed to sort pred_ranks
        s = count_inversions(pred_ranks)

        total_inversions += s
        total_pairs += n * (n - 1)

    if total_pairs == 0:
        return 0.0

    k = 1.0 - 4.0 * (total_inversions / total_pairs)
    return k


def save_checkpoint(model, optimizer, epoch, score, filename="best_model.pth"):
    """
    Saves the model checkpoint.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        epoch (int): Current epoch.
        score (float): Validation score.
        filename (str): Filename for the checkpoint.
    """
    path = os.path.join(Config.WORKING_DIR, filename)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "epoch": epoch,
            "score": score,
        },
        path,
    )


def load_checkpoint(
    model, optimizer=None, filename="best_model.pth", device=Config.DEVICE
):
    """
    Loads a model checkpoint.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        filename (str): Filename of the checkpoint.
        device (str): Device to map the location to.

    Returns:
        float: The score recorded in the checkpoint, or 0.0 if not found.
    """
    path = os.path.join(Config.WORKING_DIR, filename)
    if not os.path.exists(path):
        return 0.0

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and checkpoint["optimizer_state_dict"]:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint.get("score", 0.0)
