import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from library.config import CLIPPING_EPSILON


def calculate_log_loss(y_true, y_pred, class_labels):
    """
    Calculates the multi-class log loss after rescaling and clipping probabilities.

    This function mimics the competition's scoring mechanism:
    1. Rescales rows of y_pred to sum to 1.
    2. Clips probabilities to [eps, 1-eps].
    3. Computes log loss.

    Args:
        y_true (array-like): Ground truth labels (strings or indices).
        y_pred (array-like): Predicted probabilities matrix (n_samples, n_classes).
        class_labels (list): List of class names corresponding to the columns of y_pred.
                             This is crucial for aligning string labels in y_true.

    Returns:
        float: The calculated log loss.
    """
    # Ensure numpy array
    y_pred = np.array(y_pred)

    # 1. Rescale: Each row divided by row sum
    # The competition metric states probabilities are rescaled prior to scoring
    row_sums = y_pred.sum(axis=1)
    # Handle zero sums to avoid NaN (though unlikely with proper models)
    row_sums[row_sums == 0] = 1.0
    y_pred_rescaled = y_pred / row_sums[:, np.newaxis]

    # 2. Clip: max(min(p, 1-eps), eps)
    # This avoids infinite log loss penalties
    y_pred_clipped = np.clip(y_pred_rescaled, CLIPPING_EPSILON, 1.0 - CLIPPING_EPSILON)

    # 3. Calculate Log Loss
    # We pass 'labels' to ensure correct mapping between y_true values and y_pred columns
    score = log_loss(y_true, y_pred_clipped, labels=class_labels)

    return score


def save_submission(ids, class_labels, probabilities, output_path):
    """
    Saves the predictions to a CSV file in the required format.

    Format:
    id,Class1,Class2,...
    123,0.1,0.0,...

    Args:
        ids (array-like): List or array of image IDs.
        class_labels (list): List of class names corresponding to probability columns.
        probabilities (array-like): Matrix of predicted probabilities.
        output_path (str): File path to save the CSV.
    """
    # Create DataFrame
    df = pd.DataFrame(probabilities, columns=class_labels)

    # Insert ID column at the beginning
    df.insert(0, "id", ids)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
