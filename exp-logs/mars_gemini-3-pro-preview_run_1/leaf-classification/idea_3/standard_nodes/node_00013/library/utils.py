import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from library import config


def calculate_log_loss(y_true, y_pred):
    """
    Calculates the multi-class log loss metric with specific clipping and normalization
    as defined in the competition evaluation criteria.

    Args:
        y_true (np.array): Ground truth labels (integer indices).
        y_pred (np.array): Predicted probabilities of shape (n_samples, n_classes).

    Returns:
        float: The calculated log loss.
    """
    # Retrieve epsilon from config
    eps = config.LOG_LOSS_EPSILON

    # 1. Clip probabilities to [eps, 1-eps] to avoid extremes of the log function
    # Formula: max(min(p, 1-10^-15), 10^-15)
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)

    # 2. Rescale probabilities so each row sums to 1
    # This mimics the server-side evaluation behavior
    row_sums = y_pred_clipped.sum(axis=1)
    y_pred_norm = y_pred_clipped / row_sums[:, np.newaxis]

    # 3. Calculate Log Loss
    # We explicitly provide labels to ensure sklearn knows the full set of classes
    # even if y_true in this batch doesn't cover all of them.
    labels = np.arange(y_pred.shape[1])
    loss = log_loss(y_true, y_pred_norm, labels=labels)

    return loss


def save_submission(ids, probabilities, class_names, output_path=config.SUBMISSION_CSV):
    """
    Generates and saves the submission CSV file in the required format.

    Format:
    id, Species1, Species2, ...
    1, 0.1, 0.0, ...

    Args:
        ids (np.array or list): Sequence of image IDs corresponding to the predictions.
        probabilities (np.array): Prediction matrix of shape (n_samples, n_classes).
        class_names (list): List of species names corresponding to the columns of probabilities.
        output_path (str): File path where the submission CSV will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create the DataFrame
    df = pd.DataFrame(probabilities, columns=class_names)

    # Insert the 'id' column at the start
    df.insert(0, config.ID_COL, ids)

    # Save to CSV without the index
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
