import random
import numpy as np
import torch
from scipy.ndimage import median_filter
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def levenshtein_distance(hyp, ref):
    """
    Computes the Levenshtein edit distance between two sequences.

    Args:
        hyp (list): The hypothesis sequence (predicted gesture IDs).
        ref (list): The reference sequence (ground truth gesture IDs).

    Returns:
        int: The minimum number of single-element edits (insertions, deletions, substitutions)
             required to change hyp into ref.
    """
    n = len(hyp)
    m = len(ref)

    # Initialize DP matrix
    # dp[i][j] stores distance between hyp[:i] and ref[:j]
    dp = np.zeros((n + 1, m + 1), dtype=int)

    # Base cases: transforming empty string to/from non-empty
    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j

    # Fill DP matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if hyp[i - 1] == ref[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,  # Deletion
                dp[i, j - 1] + 1,  # Insertion
                dp[i - 1, j - 1] + cost,  # Substitution
            )

    return dp[n, m]


def compute_normalized_levenshtein(predictions, ground_truths):
    """
    Computes the global normalized Levenshtein error rate for a set of predictions.
    Metric = Sum(Levenshtein Distances) / Sum(Ground Truth Lengths).

    Args:
        predictions (list of lists): List of predicted gesture sequences.
        ground_truths (list of lists): List of ground truth gesture sequences.

    Returns:
        float: The normalized error rate.
    """
    total_dist = 0
    total_len = 0

    for p, g in zip(predictions, ground_truths):
        total_dist += levenshtein_distance(p, g)
        total_len += len(g)

    if total_len == 0:
        return 0.0

    return total_dist / total_len


def decode_predictions_rle(predictions):
    """
    Decodes frame-wise predictions into an ordered list of gestures using
    Median Filtering and Run-Length Encoding (RLE).

    Applies the following post-processing steps:
    1. Argmax (if input is probabilities).
    2. Median Filter (window size from Config).
    3. RLE segmentation.
    4. Filter out background class (0).
    5. Filter out segments shorter than Config.MIN_SEGMENT_LENGTH.

    Args:
        predictions (np.ndarray or torch.Tensor): Frame-wise predictions.
            Can be shape (T, NumClasses) [logits/probs] or (T,) [indices].

    Returns:
        list: Ordered list of integer gesture IDs.
    """
    # Handle Tensor input
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()

    # Convert logits/probs to indices if necessary
    if predictions.ndim > 1:
        indices = np.argmax(predictions, axis=1)
    else:
        indices = predictions

    # Apply Median Filter to smooth noise
    # mode='nearest' extends the boundary values
    filtered_indices = median_filter(
        indices, size=Config.MEDIAN_FILTER_KERNEL, mode="nearest"
    )

    if len(filtered_indices) == 0:
        return []

    # Run-Length Encoding Logic
    # Identify indices where the value changes
    # np.where returns indices where condition is true.
    # If arr[i] != arr[i+1], change happens after i.
    change_indices = np.where(filtered_indices[:-1] != filtered_indices[1:])[0] + 1

    # Define segment boundaries: [0, change_1, change_2, ..., length]
    split_indices = np.concatenate(([0], change_indices, [len(filtered_indices)]))

    gesture_list = []

    for i in range(len(split_indices) - 1):
        start = split_indices[i]
        end = split_indices[i + 1]
        length = end - start
        label = filtered_indices[start]

        # Step 4: Filter Background
        if label == Config.LABEL_MAP["background"]:
            continue

        # Step 5: Filter Short Segments
        if length < Config.MIN_SEGMENT_LENGTH:
            continue

        gesture_list.append(int(label))

    return gesture_list
