import numpy as np
from scipy import sparse
from sklearn.metrics import f1_score
import warnings

# Set fixed random seed
np.random.seed(42)

# Suppress warnings (e.g. zero division in f1 score for empty predictions)
warnings.filterwarnings("ignore")


def compute_f1_score(y_true, y_pred):
    """
    Calculates the Mean F1-Score (samples average).

    Args:
        y_true (sparse matrix or np.array): Ground truth binary labels.
        y_pred (sparse matrix or np.array): Predicted binary labels.

    Returns:
        float: The Mean F1-Score.
    """
    # Calculate F1 score with 'samples' average for multi-label classification
    return f1_score(y_true, y_pred, average="samples", zero_division=0)


def optimize_threshold(y_true, y_scores, thresholds=None):
    """
    Iterates through potential probability thresholds on the validation set
    to find the value that maximizes the F1-Score.

    Args:
        y_true (sparse matrix or np.array): Ground truth binary labels.
        y_scores (sparse matrix or np.array): Predicted probabilities/scores.
        thresholds (list, optional): List of thresholds to evaluate.
                                     If None, defaults to [0.1, 0.15, ..., 0.5].

    Returns:
        float: The best threshold found.
        float: The best F1-Score achieved.
    """
    if thresholds is None:
        # Default range from 0.1 to 0.5 with step 0.05
        thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    best_threshold = 0.0
    best_f1 = -1.0

    # Ensure y_scores is in a format suitable for efficient comparison
    # If sparse, operations like (sparse_matrix > value) return a sparse boolean matrix

    for thresh in thresholds:
        # Binarize predictions based on the current threshold
        # .astype(int) converts boolean to 0/1 integers
        y_pred = (y_scores > thresh).astype(int)

        # Compute F1 Score
        score = compute_f1_score(y_true, y_pred)

        # Print full precision as requested
        print(f"Threshold: {thresh}, F1-Score: {score}")

        # Update best score
        if score > best_f1:
            best_f1 = score
            best_threshold = thresh

    return best_threshold, best_f1
