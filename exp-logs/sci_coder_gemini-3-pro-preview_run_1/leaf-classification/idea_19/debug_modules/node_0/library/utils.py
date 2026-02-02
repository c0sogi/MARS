import os
import pandas as pd
import numpy as np
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    FEATURE_COLS,
    ID_COL,
    TARGET_COL,
    SUBMISSION_PATH,
    EPSILON,
)


def load_data(split="train"):
    """
    Loads the dataset for the specified split using the metadata CSVs.

    Args:
        split (str): One of 'train', 'val', or 'test'.

    Returns:
        X (pd.DataFrame): The feature matrix with columns ordered according to FEATURE_COLS.
        y (pd.Series or None): The target variable (species) for 'train'/'val', or None for 'test'.
        ids (pd.Series): The image identifiers.
    """
    if split == "train":
        path = TRAIN_METADATA_PATH
    elif split == "val":
        path = VAL_METADATA_PATH
    elif split == "test":
        path = TEST_METADATA_PATH
    else:
        raise ValueError(
            f"Invalid split argument: {split}. Expected 'train', 'val', or 'test'."
        )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at: {path}")

    # Read CSV
    df = pd.read_csv(path)

    # Validation: Check for ID column
    if ID_COL not in df.columns:
        raise ValueError(f"ID column '{ID_COL}' missing from {path}")
    ids = df[ID_COL]

    # Validation: Check for Feature columns
    # We enforce strict column ordering based on config to prevent permutation errors
    missing_features = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_features:
        raise ValueError(
            f"The following feature columns are missing in {path}: {missing_features[:5]}..."
        )

    X = df[FEATURE_COLS]

    # Extract Target if applicable
    y = None
    if split in ["train", "val"]:
        if TARGET_COL not in df.columns:
            raise ValueError(f"Target column '{TARGET_COL}' missing from {path}")
        y = df[TARGET_COL]

    return X, y, ids


def save_submission(ids, probabilities, class_names, output_path=SUBMISSION_PATH):
    """
    Formats and saves the submission file with clipped probabilities.

    Args:
        ids (array-like): Sequence of image IDs.
        probabilities (numpy.ndarray): Matrix of predicted probabilities (n_samples, n_classes).
        class_names (list): List of class names corresponding to the columns of probabilities.
        output_path (str): File path to save the CSV. Defaults to config path.
    """
    # Input validation
    if len(ids) != probabilities.shape[0]:
        raise ValueError(
            f"Mismatch between ID count ({len(ids)}) and probability rows ({probabilities.shape[0]})"
        )

    if probabilities.shape[1] != len(class_names):
        raise ValueError(
            f"Mismatch between probability columns ({probabilities.shape[1]}) and class names ({len(class_names)})"
        )

    # Clip probabilities to [EPSILON, 1-EPSILON] to avoid infinite log loss
    # The metric requires: max(min(p, 1-10^-15), 10^-15)
    clipped_probs = np.clip(probabilities, EPSILON, 1.0 - EPSILON)

    # Construct DataFrame
    submission_df = pd.DataFrame(clipped_probs, columns=class_names)
    submission_df.insert(0, ID_COL, ids)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
