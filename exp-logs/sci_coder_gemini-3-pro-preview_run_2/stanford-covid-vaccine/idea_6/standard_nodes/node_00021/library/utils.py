import os
import ast
import numpy as np
import torch
from library.config import Config


def parse_structure_pairs(structure):
    """
    Parses dot-bracket structure to find pairs.
    Returns a mapping {index: paired_index}. Unpaired indices are not in the dict.

    Args:
        structure (str): A string representing RNA secondary structure in dot-bracket notation.

    Returns:
        dict: A dictionary mapping paired base indices.
    """
    pairs = {}
    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                start = stack.pop()
                pairs[start] = i
                pairs[i] = start
    return pairs


def one_hot_encode(idx, num_classes):
    """
    One-hot encodes a class index.

    Args:
        idx (int): The index of the class.
        num_classes (int): The total number of classes.

    Returns:
        np.ndarray: A float32 numpy array of shape (num_classes,).
    """
    vec = np.zeros(num_classes, dtype=np.float32)
    if 0 <= idx < num_classes:
        vec[idx] = 1.0
    return vec


def parse_list_column(x):
    """
    Safely parses a stringified list from CSV (e.g., "[0.1, 0.2, ...]") into a numpy array.

    Args:
        x (str): The string representation of the list.

    Returns:
        np.ndarray: A float32 numpy array containing the parsed values. Returns empty array on failure.
    """
    try:
        return np.array(ast.literal_eval(x), dtype=np.float32)
    except Exception:
        return np.array([], dtype=np.float32)


def mcrmse_loss(pred, target, mask=None):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).
    The metric is calculated only for the scored columns:
    reactivity (0), deg_Mg_pH10 (1), and deg_Mg_50C (3).

    Args:
        pred (torch.Tensor): Predicted values of shape [Batch, Seq, 5].
        target (torch.Tensor): Ground truth values of shape [Batch, Seq, 5].
        mask (torch.Tensor, optional): Mask of shape [Batch, Seq] or broadcastable.
                                       1.0 for valid positions, 0.0 for ignored.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Scored indices based on competition metric: reactivity, deg_Mg_pH10, deg_Mg_50C
    scored_indices = [0, 1, 3]

    loss = 0.0
    count = 0

    for idx in scored_indices:
        p = pred[:, :, idx]
        t = target[:, :, idx]

        mse = (p - t) ** 2

        if mask is not None:
            # Apply mask
            mse = mse * mask
            # Calculate RMSE over valid positions: sqrt(sum(error^2) / count_valid)
            # Add epsilon to denominator to prevent division by zero
            rmse = torch.sqrt(mse.sum() / (mask.sum() + 1e-8))
        else:
            rmse = torch.sqrt(mse.mean())

        loss += rmse
        count += 1

    return loss / count
