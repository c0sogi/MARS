import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    SEED,
)


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to the value in config.py.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def levenshtein_distance(hyp, ref):
    """
    Computes the Levenshtein distance between two sequences.

    Args:
        hyp (list): The predicted sequence of labels (hypothesis).
        ref (list): The ground truth sequence of labels (reference).

    Returns:
        int: The edit distance.
    """
    n = len(hyp)
    m = len(ref)

    # Initialize matrix of size (n+1) x (m+1)
    # distance[i][j] is the distance between hyp[:i] and ref[:j]
    distance = np.zeros((n + 1, m + 1), dtype=int)

    # Initialize first row and column
    for i in range(n + 1):
        distance[i][0] = i
    for j in range(m + 1):
        distance[0][j] = j

    # Compute distances
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if hyp[i - 1] == ref[j - 1]:
                cost = 0
            else:
                cost = 1

            distance[i][j] = min(
                distance[i - 1][j] + 1,  # Deletion
                distance[i][j - 1] + 1,  # Insertion
                distance[i - 1][j - 1] + cost,  # Substitution
            )

    return distance[n][m]


def compute_levenshtein_score(predictions, ground_truths):
    """
    Computes the normalized Levenshtein score for a batch of predictions.
    Score = (Sum of Levenshtein Distances) / (Total Number of Ground Truth Gestures)

    Args:
        predictions (list of lists): List of predicted label sequences.
        ground_truths (list of lists): List of ground truth label sequences.

    Returns:
        float: The normalized error rate.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have the same length.")

    total_distance = 0
    total_ref_length = 0

    for hyp, ref in zip(predictions, ground_truths):
        # Ensure inputs are lists
        hyp = list(hyp) if not isinstance(hyp, list) else hyp
        ref = list(ref) if not isinstance(ref, list) else ref

        dist = levenshtein_distance(hyp, ref)
        total_distance += dist
        total_ref_length += len(ref)

    if total_ref_length == 0:
        return 0.0 if total_distance == 0 else float("inf")

    return total_distance / total_ref_length


def load_metadata(split="train"):
    """
    Loads the metadata CSV for the specified split and processes the label column.

    Args:
        split (str): One of 'train', 'val', or 'test'.

    Returns:
        pd.DataFrame: DataFrame containing the metadata with 'labels' column parsed into lists.
    """
    if split == "train":
        path = TRAIN_METADATA_PATH
    elif split == "val":
        path = VAL_METADATA_PATH
    elif split == "test":
        path = TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")

    df = pd.read_csv(path)

    # Parse the space-separated labels string into a list of integers
    # Handle NaN or empty strings gracefully
    def parse_labels(label_str):
        if pd.isna(label_str) or str(label_str).strip() == "":
            return []
        try:
            return [int(x) for x in str(label_str).strip().split()]
        except ValueError:
            return []

    if "labels" in df.columns:
        df["labels"] = df["labels"].apply(parse_labels)
    else:
        # If labels column is missing (shouldn't happen based on metadata gen), create empty lists
        df["labels"] = [[] for _ in range(len(df))]

    return df
