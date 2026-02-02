import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python's random, numpy, and os environments.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def clip_probabilities(preds):
    """
    Clips probabilities to the range [1e-15, 1-1e-15] to avoid log(0) errors
    in the log_loss metric and to strictly follow the competition's scoring formula.

    Args:
        preds (np.ndarray): The array of predicted probabilities.

    Returns:
        np.ndarray: The clipped probabilities.
    """
    epsilon = 1e-15
    # As per task description: max(min(p, 1-10^-15), 10^-15)
    return np.clip(preds, epsilon, 1 - epsilon)


def score_predictions(y_true, y_pred, labels=None):
    """
    Calculates the Multi-class Log Loss, replicating the competition's scoring logic.
    This includes row-wise normalization (rescaling) and clipping prior to scoring.

    Args:
        y_true (array-like): True labels (1D array of class indices/strings or 2D one-hot).
        y_pred (array-like): Predicted probabilities (2D array).
        labels (list, optional): List of class labels to index the prediction matrix.

    Returns:
        float: The calculated log loss.
    """
    y_pred = np.array(y_pred, dtype=float)

    # The submitted probabilities are rescaled prior to being scored
    # (each row is divided by the row sum).
    row_sums = y_pred.sum(axis=1)
    # Handle rows that sum to 0 to avoid division by zero (though unlikely with valid models)
    row_sums[row_sums == 0] = 1.0
    y_pred_scaled = y_pred / row_sums[:, np.newaxis]

    # Apply clipping as specified in the metric description
    y_pred_clipped = clip_probabilities(y_pred_scaled)

    return log_loss(y_true, y_pred_clipped, labels=labels)


def create_submission(ids, predictions, class_names, output_path="submission.csv"):
    """
    Creates a submission CSV file in the required format.

    Args:
        ids (array-like): 1D array or list of image IDs.
        predictions (np.ndarray): 2D array of probabilities (shape: [n_samples, n_classes]).
        class_names (list): List of species names corresponding to the columns of predictions.
        output_path (str): Path to save the submission file.
    """
    # Ensure predictions are float
    predictions = predictions.astype(float)

    # Create DataFrame with class names as headers
    df = pd.DataFrame(predictions, columns=class_names)

    # Insert 'id' column at the beginning
    df.insert(0, "id", ids)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save to CSV without index
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
