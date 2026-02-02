import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from library.config import SUBMISSION_DIR


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and environment variables.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    # Note: If PyTorch or TensorFlow were used here, their seeds would also be set.
    # Since this is a utility module, we stick to standard libraries.


def calculate_log_loss(y_true, y_pred, eps=1e-15):
    """
    Calculates the multi-class log loss metric as defined in the competition.

    The metric involves:
    1. Rescaling rows so they sum to 1.
    2. Clipping probabilities to [eps, 1-eps].
    3. Calculating the negative log likelihood.

    Args:
        y_true (array-like): Ground truth labels (n_samples,). Can be class indices or one-hot.
        y_pred (array-like): Predicted probabilities (n_samples, n_classes).
        eps (float): Epsilon value for clipping. Defaults to 1e-15.

    Returns:
        float: The calculated log loss.
    """
    # Ensure numpy array
    y_pred = np.array(y_pred, dtype=np.float64)

    # 1. Rescale rows to sum to 1 (as per competition metric description)
    # Handle potential division by zero if a row sums to 0 (though unlikely with valid models)
    row_sums = y_pred.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1.0
    y_pred_norm = y_pred / row_sums

    # 2. Clip probabilities
    # The competition specifies max(min(p, 1-10^-15), 10^-15)
    y_pred_clipped = np.clip(y_pred_norm, eps, 1 - eps)

    # 3. Calculate Log Loss
    # sklearn's log_loss handles the log calculation and averaging
    score = log_loss(y_true, y_pred_clipped, labels=list(range(y_pred.shape[1])))

    return score


def create_submission_file(ids, probabilities, class_names, output_path=None):
    """
    Formats the predictions into a pandas DataFrame and saves it as a CSV file
    matching the competition submission format.

    Args:
        ids (array-like): List or array of image IDs.
        probabilities (array-like): Matrix of predicted probabilities (n_samples, n_classes).
        class_names (list): List of species names corresponding to the columns of probabilities.
        output_path (str, optional): Full path to save the CSV. If None, saves to default SUBMISSION_DIR.

    Returns:
        pd.DataFrame: The created submission dataframe.
    """
    # Ensure inputs are correct types
    ids = np.array(ids).flatten()
    probabilities = np.array(probabilities)

    if probabilities.shape[1] != len(class_names):
        raise ValueError(
            f"Number of probability columns ({probabilities.shape[1]}) "
            f"does not match number of class names ({len(class_names)})."
        )

    if len(ids) != probabilities.shape[0]:
        raise ValueError(
            f"Number of IDs ({len(ids)}) does not match number of predictions ({probabilities.shape[0]})."
        )

    # Create DataFrame
    submission_df = pd.DataFrame(probabilities, columns=class_names)
    submission_df.insert(0, "id", ids)

    # Define output path
    if output_path is None:
        os.makedirs(SUBMISSION_DIR, exist_ok=True)
        output_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    else:
        # Ensure directory exists for custom path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(output_path, index=False)

    return submission_df
