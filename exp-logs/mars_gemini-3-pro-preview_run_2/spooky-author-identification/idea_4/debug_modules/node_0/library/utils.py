import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from library.config import SEED, set_seed


def seed_everything(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Wraps the configuration's set_seed function to avoid re-implementation.

    Args:
        seed (int): The seed value to set. Defaults to global SEED.
    """
    set_seed(seed)


def clip_probabilities(probs):
    """
    Clips probabilities to the range [10^-15, 1 - 10^-15] to avoid extremes of the log function,
    as specified in the metric description.

    Args:
        probs (np.ndarray): Array of predicted probabilities.

    Returns:
        np.ndarray: Clipped probabilities.
    """
    epsilon = 1e-15
    return np.clip(probs, epsilon, 1 - epsilon)


def calculate_log_loss(y_true, y_pred):
    """
    Calculates the Multi-class logarithmic loss using the clipped probabilities.

    Args:
        y_true (array-like): True labels (encoded as integers).
        y_pred (array-like): Predicted probabilities.

    Returns:
        float: The log loss value.
    """
    # Clip probabilities to ensure numerical stability and match evaluation metric
    y_pred_clipped = clip_probabilities(y_pred)

    # Calculate log loss
    # We assume y_true contains integer labels corresponding to the column indices of y_pred
    return log_loss(y_true, y_pred_clipped)


def save_submission(ids, probs, class_names, output_path):
    """
    Generates and saves the submission CSV file in the required format.

    Format:
    id,EAP,HPL,MWS
    id07943,0.33,0.33,0.33
    ...

    Args:
        ids (list or np.array): Sequence of IDs for the test set.
        probs (np.ndarray): Matrix of predicted probabilities (N_samples, N_classes).
        class_names (list): List of class names corresponding to the columns of probs.
        output_path (str): File path where the submission CSV will be saved.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame
    submission_df = pd.DataFrame(probs, columns=class_names)

    # Insert 'id' as the first column
    submission_df.insert(0, "id", ids)

    # Save to CSV without the index
    submission_df.to_csv(output_path, index=False)
