import os
import random
import numpy as np
import torch
import nltk


def set_seed(seed):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_label_string(label_str):
    """
    Parses a space-separated string of gesture labels into a list of integers.

    Args:
        label_str (str or float/nan): The string containing labels (e.g., "2 12 3").

    Returns:
        list[int]: A list of gesture IDs. Returns an empty list if input is NaN or empty.
    """
    if label_str is None:
        return []

    # Handle NaN (pandas often reads empty cells as NaN/float)
    if isinstance(label_str, float) and np.isnan(label_str):
        return []

    s = str(label_str).strip()
    if not s:
        return []

    try:
        return [int(x) for x in s.split()]
    except ValueError:
        return []


def compute_levenshtein(predicted_seqs, truth_seqs):
    """
    Computes the Levenshtein error rate for the dataset.

    Metric = (Sum of Levenshtein Distances) / (Total Number of Ground Truth Gestures)

    Args:
        predicted_seqs (list[list[int]]): List of predicted gesture sequences.
        truth_seqs (list[list[int]]): List of ground truth gesture sequences.

    Returns:
        float: The calculated error rate. Returns 0.0 if there are no ground truth gestures.
    """
    total_distance = 0
    total_truth_length = 0

    for p_seq, t_seq in zip(predicted_seqs, truth_seqs):
        # nltk.edit_distance works with lists of hashable items (integers are hashable)
        dist = nltk.edit_distance(p_seq, t_seq)
        total_distance += dist
        total_truth_length += len(t_seq)

    if total_truth_length == 0:
        return 0.0

    return total_distance / total_truth_length


def save_checkpoint(state, filename):
    """
    Saves the model and optimizer state to a file.

    Args:
        state (dict): Dictionary containing model_state_dict, optimizer_state_dict, etc.
        filename (str): Path to save the checkpoint.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None):
    """
    Loads a checkpoint into the model and optional optimizer.

    Args:
        filename (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.

    Returns:
        dict: The full checkpoint dictionary (useful for retrieving epoch, best_loss, etc.).
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    checkpoint = torch.load(filename, map_location=torch.device("cpu"))

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
