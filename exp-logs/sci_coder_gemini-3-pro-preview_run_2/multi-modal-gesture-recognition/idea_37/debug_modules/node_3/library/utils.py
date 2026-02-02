import os
import random
import numpy as np
import torch
import nltk


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def create_padding_mask(lengths, max_len=None):
    """
    Creates a binary padding mask for variable-length sequences.
    Returns a FloatTensor where 1.0 indicates a valid position and 0.0 indicates padding.

    Args:
        lengths (torch.Tensor): Tensor containing sequence lengths of shape (batch_size,).
        max_len (int, optional): Maximum sequence length. If None, derived from the max value in lengths.

    Returns:
        torch.Tensor: Mask of shape (batch_size, max_len) with 1.0 for valid and 0.0 for padding.
    """
    if max_len is None:
        max_len = lengths.max().item()

    # Create a range tensor [0, 1, ..., max_len-1]
    # Shape: (1, max_len)
    indices = torch.arange(max_len, device=lengths.device).unsqueeze(0)

    # Expand lengths to (batch_size, 1) for broadcasting
    lengths_expanded = lengths.unsqueeze(1)

    # Create mask: True where index < length
    # Convert to float: 1.0 for valid, 0.0 for padding
    mask = (indices < lengths_expanded).float()

    return mask


def compute_levenshtein(predictions, targets):
    """
    Computes the Levenshtein distance error rate.

    Metric = Sum(Levenshtein(pred_seq, target_seq)) / Sum(len(target_seq))

    Args:
        predictions (list of list of int): List of predicted gesture label sequences.
        targets (list of list of int): List of ground truth gesture label sequences.

    Returns:
        float: The calculated error rate (Levenshtein distance normalized by total target length).
    """
    total_distance = 0
    total_target_length = 0

    for pred_seq, target_seq in zip(predictions, targets):
        # nltk.edit_distance calculates the Levenshtein distance between two sequences
        # It works for lists of integers as well as strings
        dist = nltk.edit_distance(pred_seq, target_seq)
        total_distance += dist
        total_target_length += len(target_seq)

    # Avoid division by zero if the dataset is empty or has no gestures
    if total_target_length == 0:
        return 0.0

    return total_distance / total_target_length
