import os
import numpy as np
import pandas as pd
from library.config import SUBMISSION_FILE_PATH


def calculate_log_loss(y_true, y_pred):
    """
    Calculates the multi-class log loss metric with specific rescaling and clipping rules.

    According to the task description:
    1. Probabilities are rescaled so each row sums to 1.
    2. Probabilities are clipped to the range [1e-15, 1 - 1e-15].
    3. The negative log likelihood of the true labels is computed.

    Args:
        y_true (array-like): True class indices, shape (n_samples,).
        y_pred (array-like): Predicted probabilities, shape (n_samples, n_classes).

    Returns:
        float: The calculated log loss.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 1. Rescale: each row is divided by the row sum
    row_sums = y_pred.sum(axis=1, keepdims=True)
    # Handle potential zero sums to avoid division by zero (though unlikely in valid predictions)
    row_sums[row_sums == 0] = 1.0
    y_pred_rescaled = y_pred / row_sums

    # 2. Clip: max(min(p, 1-10^-15), 10^-15)
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred_rescaled, epsilon, 1.0 - epsilon)

    # 3. Calculate Log Loss
    n_samples = y_true.shape[0]
    # Ensure y_true are integers for indexing
    y_true = y_true.astype(int)

    # Advanced indexing to select the probability assigned to the true class for each sample
    true_class_probs = y_pred_clipped[np.arange(n_samples), y_true]

    # Compute negative mean log probability
    log_loss = -np.mean(np.log(true_class_probs))

    return log_loss


def save_submission(ids, probabilities, class_names, output_path=SUBMISSION_FILE_PATH):
    """
    Formats and saves the submission file to a CSV.

    The output file will have an 'id' column followed by columns for each class probability.

    Args:
        ids (array-like): List or array of image IDs.
        probabilities (array-like): Matrix of predicted probabilities (n_samples, n_classes).
        class_names (list): List of class names (strings) corresponding to the columns of probabilities.
        output_path (str): File path to save the submission CSV. Defaults to config path.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame
    submission_df = pd.DataFrame(probabilities, columns=class_names)

    # Insert 'id' column at the beginning
    submission_df.insert(0, "id", ids)

    # Save to CSV without the index
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path}")
