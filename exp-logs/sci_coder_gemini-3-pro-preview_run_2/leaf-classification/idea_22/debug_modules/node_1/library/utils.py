import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from library.config import Config


def calculate_log_loss(y_true, y_pred, labels=None):
    """
    Calculates the multi-class log loss according to the competition metric.

    The metric requires:
    1. Rescaling probabilities so each row sums to 1.
    2. Clipping probabilities to [1e-15, 1-1e-15].
    3. Calculating standard multi-class log loss.

    Args:
        y_true: Array-like of ground truth labels (integers or strings).
        y_pred: Array-like of predicted probabilities (shape: [n_samples, n_classes]).
        labels: List of class labels to index y_pred if y_true are strings/integers.
                This ensures the columns of y_pred map correctly to the classes in y_true.

    Returns:
        float: The calculated log loss.
    """
    # Ensure numpy array for manipulation
    y_pred = np.array(y_pred, dtype=np.float64)

    # 1. Rescale: Divide each row by the row sum
    # Handle potential zero sums to avoid NaN (though unlikely with proper models)
    row_sums = y_pred.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    y_pred = y_pred / row_sums[:, np.newaxis]

    # 2. Clip: Replace probabilities with max(min(p, 1-10^-15), 10^-15)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # 3. Calculate Log Loss
    # If labels are provided, sklearn ensures the mapping between y_true values
    # and y_pred columns is correct.
    return log_loss(y_true, y_pred, labels=labels)


def save_submission(ids, probabilities, classes, output_path=None):
    """
    Formats and saves the submission file in the required CSV format.

    Format:
    id,Class_1,Class_2,...
    1,0.1,0.0,...

    Args:
        ids: Array-like of image IDs.
        probabilities: Array-like of predicted probabilities (shape: [n_samples, n_classes]).
        classes: List of class names corresponding to the columns of probabilities.
        output_path: Path to save the CSV. If None, uses Config.SUBMISSION_FILE_PATH.
    """
    if output_path is None:
        output_path = Config.SUBMISSION_FILE_PATH

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame
    # The columns must be the class names
    df = pd.DataFrame(probabilities, columns=classes)

    # Insert the 'id' column at the start
    df.insert(0, "id", ids)

    # Save to CSV without the index
    df.to_csv(output_path, index=False)
