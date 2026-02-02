import os
import numpy as np
import pandas as pd


def clip_probabilities(predictions):
    """
    Clamps predicted probabilities to the range [1e-15, 1-1e-15] to avoid
    extremes in the log loss metric.

    Args:
        predictions (np.ndarray): Array of predicted probabilities.

    Returns:
        np.ndarray: Clipped probabilities.
    """
    # The metric definition specifies max(min(p, 1-10^-15), 10^-15)
    epsilon = 1e-15
    return np.clip(predictions, epsilon, 1 - epsilon)


def save_submission(
    predictions, test_ids, class_names, output_path="./submission/submission.csv"
):
    """
    Formats and saves the predictions to a CSV file in the required format.

    Args:
        predictions (np.ndarray): Matrix of probabilities (n_samples, n_classes).
        test_ids (list or np.ndarray): IDs corresponding to the test samples.
        class_names (list): List of species names corresponding to the columns of predictions.
        output_path (str): Path to save the submission CSV.
    """
    # Ensure predictions are clipped to avoid log loss penalties
    clipped_preds = clip_probabilities(predictions)

    # Create DataFrame
    df = pd.DataFrame(clipped_preds, columns=class_names)

    # Insert id column at the beginning
    df.insert(0, "id", test_ids)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def load_metadata(split):
    """
    Loads the metadata CSV for a specific split (train, val, or test).

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    valid_splits = ["train", "val", "test"]
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}.")

    path = f"./metadata/{split}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    return pd.read_csv(path)
