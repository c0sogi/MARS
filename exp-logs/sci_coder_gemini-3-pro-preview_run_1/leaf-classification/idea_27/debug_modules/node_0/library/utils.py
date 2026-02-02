import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from library.config import IDEA_DIR


def calculate_log_loss(y_true, y_pred):
    """
    Calculates the multi-class log loss using the specific clipping and normalization
    rules defined in the competition metric.

    Args:
        y_true (array-like): True class indices or labels (n_samples,).
        y_pred (array-like): Predicted probabilities (n_samples, n_classes).

    Returns:
        float: The calculated log loss.
    """
    # Ensure numpy array
    y_pred = np.array(y_pred)

    # 1. Rescale: The submitted probabilities are rescaled prior to being scored
    # (each row is divided by the row sum).
    row_sums = y_pred.sum(axis=1, keepdims=True)
    # Avoid division by zero if a row sums to 0 (though unlikely in valid predictions)
    row_sums[row_sums == 0] = 1.0
    y_pred_norm = y_pred / row_sums

    # 2. Clip: Predicted probabilities are replaced with max(min(p, 1-10^-15), 10^-15)
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred_norm, epsilon, 1 - epsilon)

    # 3. Calculate Log Loss
    # labels parameter is important if y_true doesn't contain all classes in y_pred columns
    # We assume y_pred columns correspond to sorted unique classes of the problem
    score = log_loss(y_true, y_pred_clipped)

    return score


def save_submission(
    ids, probas, class_names, output_dir="./submission", filename="submission.csv"
):
    """
    Formats and saves the submission file.

    Args:
        ids (array-like): Vector of image IDs.
        probas (array-like): Matrix of predicted probabilities (n_samples, n_classes).
        class_names (list): List of class names corresponding to the columns of probas.
        output_dir (str): Directory to save the submission file.
        filename (str): Name of the CSV file.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create DataFrame
    submission_df = pd.DataFrame(probas, columns=class_names)
    submission_df.insert(0, "id", ids)

    # Save to CSV
    output_path = os.path.join(output_dir, filename)
    submission_df.to_csv(output_path, index=False)

    print(f"Submission saved to: {output_path}")
    print(f"Submission shape: {submission_df.shape}")
