import numpy as np
from sklearn.metrics import matthews_corrcoef, confusion_matrix


def calculate_mcc(y_true, y_pred):
    """
    Calculates the Matthews Correlation Coefficient between ground truth and predictions.

    Args:
        y_true (np.array): Ground truth binary labels.
        y_pred (np.array): Predicted binary labels.

    Returns:
        float: The MCC score.
    """
    return matthews_corrcoef(y_true, y_pred)


def optimize_threshold(y_true, y_pred_proba, step=0.01):
    """
    Iterates through probability thresholds to find the cutoff that maximizes MCC.

    Args:
        y_true (np.array): Ground truth binary labels.
        y_pred_proba (np.array): Predicted probabilities (0 to 1).
        step (float): Step size for threshold iteration. Default is 0.01.

    Returns:
        tuple: (best_threshold, best_mcc)
    """
    best_mcc = -1.0
    best_threshold = 0.5

    # Define search space: 0.01 to 0.99 usually covers the relevant range
    # We use np.arange ensuring we don't include 0 or 1 exactly to avoid trivial all-0/all-1 edge cases
    # if they aren't handled well, though MCC handles them (returns 0).
    thresholds = np.arange(step, 1.0, step)

    # Iterate to find best threshold
    for threshold in thresholds:
        # Binarize predictions
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Calculate MCC
        mcc = matthews_corrcoef(y_true, y_pred)

        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = threshold

    # Print full precision as requested
    print(f"Optimization Result - Best MCC: {best_mcc}")
    print(f"Optimization Result - Best Threshold: {best_threshold}")

    return best_threshold, best_mcc


def get_detailed_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Computes a dictionary of metrics for a specific threshold.

    Args:
        y_true (np.array): Ground truth binary labels.
        y_pred_proba (np.array): Predicted probabilities.
        threshold (float): Decision threshold.

    Returns:
        dict: Dictionary containing MCC, Precision, Recall, F1, Accuracy.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    mcc = matthews_corrcoef(y_true, y_pred)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "mcc": mcc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": threshold,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
