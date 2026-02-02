import os
import logging
import torch
import numpy as np
from library.config import Config, set_seed


def get_logger(name=__name__):
    """
    Creates and configures a logger for console output.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger


def levenshtein_distance(seq1, seq2):
    """
    Calculates the Levenshtein distance between two sequences using dynamic programming.

    Args:
        seq1 (list or np.array): First sequence of items.
        seq2 (list or np.array): Second sequence of items.

    Returns:
        float: The Levenshtein distance.
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))

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
    return matrix[size_x - 1, size_y - 1]


def compute_levenshtein(predictions, targets):
    """
    Computes the Levenshtein metric: Total Distance / Total Ground Truth Length.

    Args:
        predictions (list of list of int): Predicted sequences of gesture IDs.
        targets (list of list of int): Ground truth sequences of gesture IDs.

    Returns:
        float: The calculated error rate (can be > 1.0).
    """
    total_distance = 0
    total_length = 0

    for pred, target in zip(predictions, targets):
        # Ensure inputs are lists
        p = list(pred) if not isinstance(pred, list) else pred
        t = list(target) if not isinstance(target, list) else target

        dist = levenshtein_distance(p, t)
        total_distance += dist
        total_length += len(t)

    if total_length == 0:
        return 0.0

    return total_distance / total_length


def save_checkpoint(model, optimizer, epoch, val_metric, path):
    """
    Saves the model checkpoint to the specified path.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        epoch (int): Current epoch.
        val_metric (float): Validation metric value.
        path (str): Path to save the checkpoint.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "val_metric": val_metric,
    }
    torch.save(state, path)


def load_checkpoint(model, path, optimizer=None, device="cpu"):
    """
    Loads a model checkpoint from the specified path.

    Args:
        model (torch.nn.Module): The model to load weights into.
        path (str): Path to the checkpoint file.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        device (str or torch.device): Device to map the checkpoint to.

    Returns:
        dict: The loaded checkpoint dictionary, or None if path doesn't exist.
    """
    if not os.path.exists(path):
        return None

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and checkpoint["optimizer_state_dict"]:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def save_submission(sample_ids, predictions, path):
    """
    Saves predictions to a CSV file in the required submission format.
    Format: SessionID,Label1,Label2,...

    Args:
        sample_ids (list of str): List of sample IDs (e.g., 'Sample00300').
        predictions (list of list of int): List of predicted gesture sequences.
        path (str): Path to save the CSV.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        for sid, pred in zip(sample_ids, predictions):
            # Convert integers to string and join by comma
            pred_str = ",".join(map(str, pred))
            line = f"{sid},{pred_str}\n"
            f.write(line)
