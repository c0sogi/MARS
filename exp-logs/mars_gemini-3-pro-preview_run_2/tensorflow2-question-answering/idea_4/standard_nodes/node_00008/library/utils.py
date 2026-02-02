import os
import random
import numpy as np
import torch
import collections


def set_seed(seed: int):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Saves the model checkpoint to the specified path.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        epoch (int): Current epoch number.
        loss (float): Current validation loss.
        path (str): File path to save the checkpoint.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "loss": loss,
    }
    torch.save(state, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path, model, optimizer=None):
    """
    Loads the model checkpoint from the specified path.

    Args:
        path (str): File path to load the checkpoint from.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.

    Returns:
        dict: The loaded checkpoint dictionary, or None if file not found.
    """
    if not os.path.exists(path):
        print(f"No checkpoint found at {path}")
        return None

    # Load to CPU to avoid CUDA OOM if loading on a different device setup
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Checkpoint loaded from {path}")
    return checkpoint


def compute_exact_match(prediction, ground_truth):
    """
    Computes Exact Match (EM) score.

    Args:
        prediction: The predicted value (string, tuple, or list).
        ground_truth: The ground truth value.

    Returns:
        float: 1.0 if match, 0.0 otherwise.
    """
    return 1.0 if prediction == ground_truth else 0.0


def _normalize_input(data):
    """
    Helper function to normalize inputs for F1 calculation.
    Converts span tuples (start, end) to list of indices.
    Converts strings to list of tokens.
    """
    if data is None:
        return []

    # Handle span tuple (start, end) - exclusive end
    if (
        isinstance(data, tuple)
        and len(data) == 2
        and isinstance(data[0], int)
        and isinstance(data[1], int)
    ):
        # Handle empty or invalid spans
        if data[0] >= data[1]:
            return []
        return list(range(data[0], data[1]))

    # Handle string input
    if isinstance(data, str):
        return data.split()

    # Handle list or array
    if isinstance(data, (list, np.ndarray)):
        return list(data)

    return [data]


def compute_f1(prediction, ground_truth):
    """
    Computes F1 score based on token or index overlap.
    Suitable for span evaluation where prediction and ground_truth are (start, end) tuples.

    Args:
        prediction: Predicted span tuple (start, end) or list of tokens.
        ground_truth: Ground truth span tuple (start, end) or list of tokens.

    Returns:
        float: The F1 score.
    """
    pred_items = _normalize_input(prediction)
    gold_items = _normalize_input(ground_truth)

    common = collections.Counter(pred_items) & collections.Counter(gold_items)
    num_same = sum(common.values())

    # If either is empty
    if len(pred_items) == 0 or len(gold_items) == 0:
        return 1.0 if len(pred_items) == len(gold_items) else 0.0

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_items)
    recall = 1.0 * num_same / len(gold_items)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1
