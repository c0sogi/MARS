import os
import random
import numpy as np
import torch
from library import config


def set_seed(seed=config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def levenshtein_distance(seq1, seq2):
    """
    Calculates the Levenshtein distance between two sequences using Dynamic Programming.

    Args:
        seq1 (list): First sequence.
        seq2 (list): Second sequence.

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
    return matrix[size_x - 1, size_y - 1]


def compute_levenshtein(predictions, targets):
    """
    Computes the normalized Levenshtein distance metric (Error Rate).
    Metric = Sum(Levenshtein Distances) / Sum(Ground Truth Lengths)

    Args:
        predictions (list of list of int): Predicted gesture sequences.
        targets (list of list of int): Ground truth gesture sequences.

    Returns:
        float: The calculated error rate.
    """
    total_distance = 0
    total_truth_length = 0

    if len(predictions) != len(targets):
        raise ValueError(
            f"Mismatch in number of predictions ({len(predictions)}) and targets ({len(targets)})"
        )

    for pred, target in zip(predictions, targets):
        dist = levenshtein_distance(pred, target)
        total_distance += dist
        total_truth_length += len(target)

    if total_truth_length == 0:
        return 0.0

    score = total_distance / total_truth_length
    return score


def save_submission(predictions, output_path=config.SUBMISSION_FILE_PATH):
    """
    Saves predictions to a file in the required format: SessionID,Label1,Label2,...

    Args:
        predictions (dict): Dictionary mapping sample_id (str) to list of gesture IDs (int).
        output_path (str): Path to save the submission file.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = []
    # Sort by sample_id for consistent output order
    sorted_ids = sorted(predictions.keys())

    for sample_id in sorted_ids:
        labels = predictions[sample_id]
        # Join labels with commas
        label_str = ",".join(map(str, labels))
        line = f"{sample_id},{label_str}"
        lines.append(line)

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Submission saved to {output_path}")
