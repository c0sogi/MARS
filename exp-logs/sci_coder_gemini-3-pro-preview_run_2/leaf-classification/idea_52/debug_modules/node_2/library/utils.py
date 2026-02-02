import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from library.config import PROB_CLIP_EPS


def clipped_log_loss(y_true, y_pred):
    """
    Calculates the multi-class log loss with specific normalization and clipping
    requirements defined in the task.

    The submitted probabilities are rescaled prior to being scored (each row is
    divided by the row sum). Predicted probabilities are replaced with
    max(min(p, 1-10^-15), 10^-15).

    Args:
        y_true (array-like): Ground truth labels (n_samples,). Can be class indices.
        y_pred (array-like): Predicted probabilities (n_samples, n_classes).

    Returns:
        float: The calculated log loss.
    """
    y_pred = np.array(y_pred, dtype=np.float64)

    # 1. Rescale probabilities: each row divided by the row sum
    row_sums = y_pred.sum(axis=1)

    # Handle potential zero sums to avoid division by zero (though unlikely with valid models)
    # If a row sums to 0, we leave it as 0 (or uniform) - but here we just avoid NaN.
    # In practice, models should output non-zero sums.
    mask_nonzero = row_sums > 0
    y_pred_norm = y_pred.copy()
    y_pred_norm[mask_nonzero] /= row_sums[mask_nonzero, np.newaxis]

    # 2. Calculate Log Loss
    # Manually clip probabilities as 'eps' argument is deprecated in sklearn 1.5+
    y_pred_clipped = np.clip(y_pred_norm, PROB_CLIP_EPS, 1 - PROB_CLIP_EPS)
    score = log_loss(y_true, y_pred_clipped)

    return score


def save_submission(ids, probabilities, class_names, output_path):
    """
    Formats and saves the predictions to a CSV file.

    Args:
        ids (array-like): List or array of image IDs.
        probabilities (array-like): Matrix of predicted probabilities (n_samples, n_classes).
        class_names (list): List of class names corresponding to the columns of probabilities.
        output_path (str): File path to save the submission CSV.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame
    # Columns must be the species names
    df = pd.DataFrame(probabilities, columns=class_names)

    # Insert 'id' as the first column
    df.insert(0, "id", ids)

    # Save to CSV without index
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
