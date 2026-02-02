import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from library.config import Config


def calculate_log_loss(y_true, y_pred):
    """
    Calculates the multi-class log loss metric as defined in the task specifications.

    The scoring process involves:
    1. Rescaling the rows of the probability matrix so they sum to 1.
    2. Clipping the probabilities to the range [1e-15, 1 - 1e-15] to avoid log(0).
    3. Computing the log loss.

    Args:
        y_true (np.ndarray): True class labels. Can be an array of integers (0 to n_classes-1)
                             or a one-hot encoded matrix.
        y_pred (np.ndarray): Predicted probabilities matrix of shape (n_samples, n_classes).

    Returns:
        float: The computed log loss value.
    """
    # Ensure predictions are a numpy array
    y_pred = np.array(y_pred)

    # 1. Rescale probabilities (Normalization)
    # The task states probabilities are rescaled prior to scoring.
    # We calculate row sums and divide.
    row_sums = y_pred.sum(axis=1, keepdims=True)
    # Handle potential division by zero (though unlikely for valid model outputs)
    row_sums[row_sums == 0] = 1.0
    y_pred_norm = y_pred / row_sums

    # 2. Clip probabilities
    # Task requirement: max(min(p, 1-10^-15), 10^-15)
    eps = Config.PROB_CLIP
    y_pred_clipped = np.clip(y_pred_norm, eps, 1.0 - eps)

    # 3. Calculate Log Loss
    # We explicitly provide labels to handle cases where a batch might not contain all classes.
    # Assuming y_pred columns are aligned with classes 0, 1, ..., K-1
    labels = np.arange(y_pred.shape[1])

    # sklearn's log_loss handles both integer labels and one-hot encoding
    # We use the clipped probabilities.
    loss = log_loss(y_true, y_pred_clipped, labels=labels)

    return loss


def create_submission(test_ids, predictions, class_names, output_path=None):
    """
    Generates a submission CSV file in the format required by the competition.

    The file will look like:
    id,Class1,Class2,...
    12,0.1,0.0,...

    Args:
        test_ids (np.ndarray): Array of IDs for the test images.
        predictions (np.ndarray): Matrix of predicted probabilities (n_test, n_classes).
        class_names (list or np.ndarray): Sequence of class names corresponding to the
                                          columns of the predictions matrix.
        output_path (str, optional): File path to save the CSV. If None, uses the path
                                     defined in Config.SUBMISSION_PATH.
    """
    if output_path is None:
        output_path = Config.SUBMISSION_PATH

    # Ensure the directory for the submission file exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Validation
    if len(test_ids) != len(predictions):
        raise ValueError(
            f"Mismatch: {len(test_ids)} IDs and {len(predictions)} prediction rows."
        )

    if predictions.shape[1] != len(class_names):
        raise ValueError(
            f"Mismatch: {predictions.shape[1]} prediction columns and {len(class_names)} class names."
        )

    # Construct DataFrame
    # The columns must be the species names
    df = pd.DataFrame(predictions, columns=class_names)

    # Insert the 'id' column at the beginning
    df.insert(0, Config.ID_COL, test_ids)

    # Save to CSV without the index
    df.to_csv(output_path, index=False)

    print(f"Submission saved to {output_path}")
