import os
import numpy as np
import pandas as pd
from library.config import SUBMISSION_DIR, CLIP_EPSILON, FLOAT_TYPE


def save_submission(ids, probabilities, class_names, filename="submission.csv"):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        ids (array-like): List or array of image IDs.
        probabilities (array-like): Matrix of predicted probabilities (N_samples, N_classes).
        class_names (list): List of class names corresponding to the columns of probabilities.
        filename (str): Name of the output file.
    """
    # Ensure output directory exists
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Ensure probabilities are numpy array with correct precision
    probs = np.array(probabilities, dtype=FLOAT_TYPE)

    # Create DataFrame
    df = pd.DataFrame(probs, columns=class_names)

    # Insert ID column at the beginning
    df.insert(0, "id", ids)

    # Construct full path
    output_path = os.path.join(SUBMISSION_DIR, filename)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def calculate_log_loss(y_true, y_pred, class_names=None):
    """
    Computes the multi-class log loss with specific clipping and normalization rules
    as defined in the competition metric.

    Rules:
    1. The submitted probabilities are rescaled prior to being scored (each row is divided by the row sum).
    2. Predicted probabilities are replaced with max(min(p, 1-10^-15), 10^-15).

    Args:
        y_true (array-like): Ground truth labels. Can be class indices (int) or class names (str).
        y_pred (array-like): Predicted probabilities (N_samples, N_classes).
        class_names (list, optional): List of class names. Required if y_true contains strings.
                                      Used to map strings to indices.

    Returns:
        float: The calculated log loss.
    """
    # Ensure predictions are float64 for precision
    y_pred = np.array(y_pred, dtype=FLOAT_TYPE)

    # 1. Rescale: each row divided by row sum
    row_sums = y_pred.sum(axis=1, keepdims=True)
    # Handle rows that sum to zero (prevent division by zero)
    row_sums[row_sums == 0] = 1.0
    y_pred_norm = y_pred / row_sums

    # 2. Clip: max(min(p, 1-1e-15), 1e-15)
    y_pred_clipped = np.clip(y_pred_norm, CLIP_EPSILON, 1.0 - CLIP_EPSILON)

    # 3. Process Ground Truth
    y_true = np.array(y_true)

    # If y_true contains strings (class names)
    if (
        y_true.dtype.kind in {"U", "S", "O"}
        and len(y_true) > 0
        and isinstance(y_true[0], str)
    ):
        if class_names is None:
            raise ValueError("class_names must be provided if y_true contains strings.")

        # Create a mapping from class name to index
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        try:
            y_true_indices = np.array(
                [class_to_idx[label] for label in y_true], dtype=int
            )
        except KeyError as e:
            raise ValueError(f"Ground truth label {e} not found in class_names.")
    else:
        # Assume inputs are already indices
        y_true_indices = np.array(y_true, dtype=int)

    # 4. Calculate Log Loss
    # Formula: -1/N * sum(log(p_true_class))
    n_samples = y_pred.shape[0]

    # Extract the probabilities corresponding to the true classes using advanced indexing
    prob_true = y_pred_clipped[np.arange(n_samples), y_true_indices]

    # Compute negative log likelihood
    loss = -np.mean(np.log(prob_true))

    return loss


def get_class_names_from_submission(sample_submission_path):
    """
    Extracts class names from the sample submission file to ensure
    the correct column order for submission.

    Args:
        sample_submission_path (str): Path to sample_submission.csv

    Returns:
        list: List of class names (strings).
    """
    if not os.path.exists(sample_submission_path):
        raise FileNotFoundError(
            f"Sample submission file not found at {sample_submission_path}"
        )

    # Read just the header to get column names
    df = pd.read_csv(sample_submission_path, nrows=0)

    # Filter out 'id' to get just the class names
    class_names = [c for c in df.columns if c != "id"]

    return class_names
