import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_lrap(y_true, y_score):
    """
    Calculates the Label-Weighted Label-Ranking Average Precision (lwlrap).

    This metric calculates the average precision for each label based on its ranking
    among all predictions, and then averages these scores across all labels.

    Args:
        y_true (np.ndarray): Binary ground truth matrix of shape (n_samples, n_classes).
        y_score (np.ndarray): Predicted probabilities matrix of shape (n_samples, n_classes).

    Returns:
        float: The overall label-weighted label-ranking average precision.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    assert (
        y_true.shape == y_score.shape
    ), "Shapes of ground truth and scores must match."

    n_samples, n_classes = y_true.shape

    # Sort scores descending to get ranking indices
    # argsort sorts ascending, so we negate y_score to sort descending
    # sort_order[i, j] contains the class index that is at rank j for sample i
    sort_order = np.argsort(-y_score, axis=1)

    # Create indices to rearrange y_true based on score ranks
    sample_indices = np.arange(n_samples)[:, None]

    # Rearrange ground truth: y_true_sorted[i, k] is 1 if the class at rank k (0-indexed) is present
    y_true_sorted = y_true[sample_indices, sort_order]

    # Calculate cumulative sum of true positives at each rank
    # cumulative_hits[i, k] = number of true labels in the top k+1 predictions for sample i
    cumulative_hits = np.cumsum(y_true_sorted, axis=1)

    # Ranks are 1, 2, ..., n_classes
    ranks = np.arange(1, n_classes + 1)

    # Precision at rank k = (number of hits in top k) / k
    precisions = cumulative_hits / ranks

    # We only care about the precisions at the ranks where the label is actually true.
    # If the label at rank k is not true, it doesn't contribute to the Average Precision for that label.
    relevant_precisions = precisions * y_true_sorted

    # Now aggregate per class. We need to sum the precisions for each specific class.
    per_class_scores = np.zeros(n_classes)

    # Iterate over each class to sum its specific precisions
    for class_id in range(n_classes):
        # Identify where this class appears in the sorted list
        # sort_order contains the original class index at each rank position
        class_positions = sort_order == class_id

        # Extract the precisions calculated at the ranks where this class was placed.
        # relevant_precisions has values > 0 only at ranks where a hit occurred.
        # By masking with class_positions, we select the precision contributions
        # specifically when the hit was for 'class_id'.
        class_contributions = relevant_precisions[class_positions]

        # Sum contributions
        score_sum = class_contributions.sum()

        # Normalize by the total number of ground truth instances for this class
        total_ground_truth = y_true[:, class_id].sum()

        if total_ground_truth > 0:
            per_class_scores[class_id] = score_sum / total_ground_truth
        else:
            per_class_scores[class_id] = 0.0

    # Calculate overall average (macro-average over classes)
    overall_lwlrap = np.mean(per_class_scores)

    return overall_lwlrap


def save_checkpoint(state, is_best, filepath):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model parameters, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filepath (str): Path to save the checkpoint.
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    torch.save(state, filepath)
