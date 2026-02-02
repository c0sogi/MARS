import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def rle_encode(mask):
    """
    Encodes a binary mask to Run-Length Encoding (RLE) format for submission.

    The metric checks that pairs are sorted, positive, and decoded pixel values are not duplicated.
    Pixels are numbered from left to right, then top to bottom (Row-Major).

    Args:
        mask (np.ndarray): Binary mask (0 or 1) of shape (H, W).

    Returns:
        str: A space-delimited string of run-length encoded pairs (start length).
    """
    # Flatten the mask in row-major order
    pixels = mask.flatten()

    # Pad with zeros at the beginning and end to detect all state changes
    # (e.g., if the mask starts or ends with 1s)
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes (0->1 or 1->0)
    # np.where returns indices in the padded array.
    # The +1 adjusts for the shift caused by slicing pixels[1:] vs pixels[:-1]
    # effectively pointing to the start of the new run.
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths: The end of a run of 1s is the start of the next run of 0s.
    # We subtract the start index from the end index to get the length.
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def calculate_fbeta(y_true, y_pred, beta=0.5, epsilon=1e-7):
    """
    Calculates the F-beta score, weighting precision higher than recall when beta < 1.

    Formula: (1 + beta^2) * (precision * recall) / ((beta^2 * precision) + recall)

    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_pred (np.ndarray): Predicted binary labels.
        beta (float): Weight of precision in the harmonic mean. Defaults to 0.5.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        float: The F-beta score.
    """
    # Ensure inputs are flat numpy arrays for global metric calculation
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    tp = (y_true * y_pred).sum()
    fp = ((1 - y_true) * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    score = (
        (1 + beta**2)
        * (precision * recall)
        / ((beta**2 * precision) + recall + epsilon)
    )

    return score


def find_best_threshold(y_true, y_pred_probs, beta=0.5, start=0.01, end=1.0, step=0.01):
    """
    Searches for the probability threshold that maximizes the F-beta score on the provided data.

    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_pred_probs (np.ndarray): Predicted probabilities (floats in [0, 1]).
        beta (float): Beta value for F-score.
        start (float): Starting threshold.
        end (float): Ending threshold.
        step (float): Step size for search.

    Returns:
        tuple: (best_threshold, best_score)
    """
    best_threshold = 0.5
    best_score = 0.0

    # Flatten data once to optimize the loop
    y_true_flat = np.asarray(y_true).flatten()
    y_pred_probs_flat = np.asarray(y_pred_probs).flatten()

    thresholds = np.arange(start, end, step)

    for thresh in thresholds:
        # Binarize predictions based on current threshold
        y_pred_bin = (y_pred_probs_flat >= thresh).astype(int)

        score = calculate_fbeta(y_true_flat, y_pred_bin, beta=beta)

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold, best_score
