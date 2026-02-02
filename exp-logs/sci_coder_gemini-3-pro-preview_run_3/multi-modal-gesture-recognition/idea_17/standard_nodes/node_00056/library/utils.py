import torch
import torch.nn as nn
import numpy as np
import random
import os
from library.config import Config


def set_seed():
    """
    Sets the random seed for reproducibility using the configuration.
    Wraps the static method provided in Config.
    """
    Config.set_seed()


def levenshtein_distance(seq1, seq2):
    """
    Calculates the Levenshtein distance between two sequences using dynamic programming.

    Args:
        seq1 (list): First sequence of items (e.g., predicted gesture IDs).
        seq2 (list): Second sequence of items (e.g., ground truth gesture IDs).

    Returns:
        int: The Levenshtein distance (minimum edits to transform seq1 to seq2).
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
    return int(matrix[size_x - 1, size_y - 1])


def compute_levenshtein(predictions, targets):
    """
    Computes the normalized Levenshtein distance metric (Error Rate).

    Metric = Sum(Levenshtein Distances) / Sum(Total Ground Truth Gestures)

    Args:
        predictions (list of lists): List of predicted gesture ID sequences.
        targets (list of lists): List of ground truth gesture ID sequences.

    Returns:
        float: The computed error rate.
    """
    total_distance = 0
    total_len = 0

    for p, t in zip(predictions, targets):
        dist = levenshtein_distance(p, t)
        total_distance += dist
        total_len += len(t)

    if total_len == 0:
        return 0.0

    return total_distance / total_len


class LogSpaceSmoothingLoss(nn.Module):
    """
    A Truncated MSE loss applied to the log-probabilities of adjacent frames
    to encourage temporal smoothness without penalizing sharp boundaries excessively.
    """

    def __init__(self, weight=Config.MSE_SMOOTHING_WEIGHT, threshold=1.0):
        """
        Args:
            weight (float): Scaling factor for the loss.
            threshold (float): Maximum value for the squared difference (truncation point).
                       Set to 1.0 to be boundary-permissive (Cite Lesson 00055).
        """
        super().__init__()
        self.weight = weight
        self.threshold = threshold

    def forward(self, log_probs):
        """
        Calculates the temporal smoothing loss.

        Args:
            log_probs (torch.Tensor): Log probabilities tensor.
                                      Supports shape (Batch, Classes, Time) or (Batch, Time, Classes).
        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Determine dimensions to calculate difference along the Time axis.
        # Config.NUM_CLASSES is used to infer the class dimension.
        if log_probs.shape[1] == Config.NUM_CLASSES:
            # Shape is (Batch, Classes, Time)
            # Calculate diff along dimension 2 (Time)
            diff = log_probs[:, :, 1:] - log_probs[:, :, :-1]
        else:
            # Shape is (Batch, Time, Classes)
            # Calculate diff along dimension 1 (Time)
            diff = log_probs[:, 1:, :] - log_probs[:, :-1, :]

        # Squared Error
        loss = diff.pow(2)

        # Truncate the error (clamp) to avoid exploding loss at valid sharp transitions
        loss = torch.clamp(loss, max=self.threshold)

        # Average over all elements and scale
        return self.weight * loss.mean()
