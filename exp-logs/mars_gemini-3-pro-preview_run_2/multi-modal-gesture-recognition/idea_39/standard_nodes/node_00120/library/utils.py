import os
import random
import numpy as np
import torch
from library.config import SEED


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Force deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_levenshtein(seq1, seq2):
    """
    Computes the Levenshtein distance between two sequences of integers.
    Args:
        seq1 (list[int]): First sequence (prediction).
        seq2 (list[int]): Second sequence (ground truth).
    Returns:
        int: The edit distance.
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y), dtype=int)

    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = matrix[x - 1, y - 1]
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,  # Deletion
                    matrix[x - 1, y - 1] + 1,  # Substitution
                    matrix[x, y - 1] + 1,  # Insertion
                )
    return int(matrix[size_x - 1, size_y - 1])


def evaluate_levenshtein_accuracy(predictions, targets):
    """
    Computes the competition metric: Sum(Levenshtein) / Total_GT_Gestures.
    Args:
        predictions (list[list[int]]): List of predicted gesture sequences.
        targets (list[list[int]]): List of ground truth gesture sequences.
    Returns:
        float: The error rate (lower is better).
    """
    total_distance = 0
    total_len = 0

    for p, t in zip(predictions, targets):
        dist = compute_levenshtein(p, t)
        total_distance += dist
        total_len += len(t)

    if total_len == 0:
        return 0.0 if total_distance == 0 else float("inf")

    return float(total_distance) / float(total_len)


def save_checkpoint(model, optimizer, epoch, val_loss, path):
    """
    Saves the model checkpoint.
    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        epoch (int): Current epoch.
        val_loss (float): Validation loss at this checkpoint.
        path (str): File path to save the checkpoint.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "val_loss": val_loss,
    }
    torch.save(state, path)


def load_checkpoint(model, optimizer, path, device="cpu"):
    """
    Loads the model checkpoint.
    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer): The optimizer to load state into (can be None).
        path (str): File path to the checkpoint.
        device (str): Device to map the checkpoint to.
    Returns:
        tuple: (start_epoch, val_loss)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint.get("epoch", 0), checkpoint.get("val_loss", float("inf"))
