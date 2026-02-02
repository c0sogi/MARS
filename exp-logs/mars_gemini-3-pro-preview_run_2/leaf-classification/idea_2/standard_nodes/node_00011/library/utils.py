import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from library.config import Config


def calculate_metric(y_true, y_pred, classes=None):
    """
    Calculates the Multi-class Log Loss metric with specific rescaling and clipping.

    Per the task description:
    1. Probabilities are rescaled (row-wise) to sum to 1.
    2. Probabilities are clipped to [1e-15, 1-1e-15].

    Args:
        y_true (array-like): Ground truth labels (strings or class indices).
        y_pred (array-like): Predicted probabilities matrix of shape (n_samples, n_classes).
        classes (list, optional): List of class names corresponding to the columns of y_pred.
                                  Required if y_true contains string labels.

    Returns:
        float: The calculated Multi-class Log Loss.
    """
    # Ensure y_pred is a numpy array
    y_pred = np.array(y_pred)

    # 1. Rescale: Divide each row by its sum to ensure valid probability distribution
    # Handle potential zero sums to avoid division by zero (though unlikely in valid models)
    row_sums = y_pred.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    y_pred_norm = y_pred / row_sums[:, np.newaxis]

    # 2. Clip: Apply the specific clipping formula max(min(p, 1-eps), eps)
    eps = Config.CLIP_EPSILON
    y_pred_clipped = np.clip(y_pred_norm, eps, 1 - eps)

    # 3. Calculate Log Loss
    # sklearn.metrics.log_loss handles the cross-entropy calculation
    # We pass the processed probabilities which mimic the leaderboard scoring mechanism
    score = log_loss(y_true, y_pred_clipped, labels=classes)

    return score


def save_submission(ids, classes, probs, output_path=Config.SUBMISSION_PATH):
    """
    Saves the predicted probabilities to a CSV file in the required format.

    Format:
    id,Acer_Capillipes,Acer_Circinatum,...
    2,0.1,0.5,...

    Args:
        ids (array-like): List or array of image IDs.
        classes (list): List of class names (column headers) corresponding to probs columns.
        probs (array-like): Matrix of predicted probabilities.
        output_path (str): File path where the submission CSV will be saved.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame
    submission_df = pd.DataFrame(probs, columns=classes)

    # Insert the 'id' column at the beginning
    submission_df.insert(0, "id", ids)

    # Save to CSV without the index
    submission_df.to_csv(output_path, index=False)
