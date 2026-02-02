import numpy as np
from sklearn.metrics import matthews_corrcoef


def calculate_mcc(y_true, y_pred):
    """
    Calculates the Matthews Correlation Coefficient (MCC) between ground truth and predictions.

    Args:
        y_true (array-like): Ground truth binary labels (0 or 1).
        y_pred (array-like): Predicted binary labels (0 or 1).

    Returns:
        float: The Matthews Correlation Coefficient.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return matthews_corrcoef(y_true, y_pred)


def optimize_threshold(y_true, y_pred_proba, num_steps=100):
    """
    Performs a grid search to find the probability threshold that maximizes the MCC score.

    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred_proba (array-like): Predicted probabilities for the positive class (contact=1).
        num_steps (int): Number of threshold steps to evaluate between 0.01 and 0.99.

    Returns:
        tuple: (best_threshold, best_mcc) where best_threshold is the float threshold
               and best_mcc is the corresponding MCC score.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)

    best_mcc = -1.0
    best_threshold = 0.5

    # Generate a range of thresholds to test.
    # We use linspace to cover the range [0.01, 0.99].
    thresholds = np.linspace(0.01, 0.99, num_steps)

    for thresh in thresholds:
        # Convert probabilities to binary predictions based on the current threshold
        y_pred_binary = (y_pred_proba >= thresh).astype(int)

        # Calculate the metric for this threshold
        score = matthews_corrcoef(y_true, y_pred_binary)

        # Update best score if current is better
        if score > best_mcc:
            best_mcc = score
            best_threshold = thresh

    return best_threshold, best_mcc
