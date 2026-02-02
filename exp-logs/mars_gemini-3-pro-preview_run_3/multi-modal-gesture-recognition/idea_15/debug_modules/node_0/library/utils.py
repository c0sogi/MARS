import os
import csv
import numpy as np
import torch
import random
from itertools import groupby
from typing import List, Dict, Union, Any

# Import Config to leverage existing configuration and seeding logic
from library.config import Config


def set_seed(seed: int = None) -> None:
    """
    Sets the random seed for reproducibility using the Config class.

    Args:
        seed (int, optional): The seed to set. If None, uses Config.SEED.
    """
    Config.seed_everything(seed)


def rle_encode(predictions: Union[List[int], np.ndarray]) -> List[int]:
    """
    Converts frame-wise predictions into a sequence of gesture IDs
    by collapsing consecutive duplicates and removing the background class.

    Args:
        predictions (List[int] or np.ndarray): Frame-wise class IDs.

    Returns:
        List[int]: Ordered list of recognized gesture IDs (excluding background 0).
    """
    # Collapse consecutive identical values
    collapsed = [k for k, g in groupby(predictions)]

    # Filter out background class (0)
    # The gesture classes are 1-20
    filtered = [int(k) for k in collapsed if k != 0]

    return filtered


def levenshtein_distance(seq1: List[int], seq2: List[int]) -> int:
    """
    Calculates the Levenshtein distance between two sequences.

    Args:
        seq1 (List[int]): First sequence (e.g., prediction).
        seq2 (List[int]): Second sequence (e.g., ground truth).

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
                    matrix[x, y - 1] + 1,  # Insertion
                    matrix[x - 1, y - 1] + 1,  # Substitution
                )

    return int(matrix[size_x - 1, size_y - 1])


def compute_normalized_levenshtein(
    predictions: List[List[int]], targets: List[List[int]]
) -> float:
    """
    Computes the competition metric: Sum of Levenshtein distances divided by
    total number of ground truth gestures.

    Args:
        predictions (List[List[int]]): List of predicted sequences.
        targets (List[List[int]]): List of ground truth sequences.

    Returns:
        float: The normalized error rate.
    """
    total_distance = 0
    total_truth_length = 0

    for pred, target in zip(predictions, targets):
        dist = levenshtein_distance(pred, target)
        total_distance += dist
        total_truth_length += len(target)

    if total_truth_length == 0:
        return 0.0

    return total_distance / total_truth_length


def save_submission(predictions: Dict[str, List[int]], output_path: str = None) -> None:
    """
    Saves predictions to a CSV file in the required format.
    Format: SessionID,Label1,Label2,...

    Args:
        predictions (Dict[str, List[int]]): Dictionary mapping sample_id to list of gesture IDs.
        output_path (str, optional): Path to save the CSV. Defaults to Config.SUBMISSION_FILE.
    """
    if output_path is None:
        output_path = Config.SUBMISSION_FILE

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Sort keys to ensure deterministic output order if needed,
        # though usually not strictly required, it's good practice.
        for sample_id in sorted(predictions.keys()):
            gestures = predictions[sample_id]
            # Row format: [SessionID, gesture_1, gesture_2, ...]
            row = [sample_id] + gestures
            writer.writerow(row)

    print(f"Submission saved to {output_path}")
