import os
import random
import numpy as np
import torch
import pandas as pd


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_submission_file(
    ids, probs, class_names, output_path="./submission/submission.csv"
):
    """
    Formats predictions and saves them to a CSV file in the required submission format.

    Args:
        ids (list or np.ndarray): Sequence of image IDs corresponding to the predictions.
        probs (np.ndarray): A 2D array of shape (n_samples, n_classes) containing the predicted probabilities.
        class_names (list): A list of strings representing the class names (column headers).
        output_path (str): The file path where the submission CSV will be saved.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Input validation
    if len(ids) != probs.shape[0]:
        raise ValueError(
            f"Mismatch between number of IDs ({len(ids)}) and number of probability rows ({probs.shape[0]})."
        )

    if len(class_names) != probs.shape[1]:
        raise ValueError(
            f"Mismatch between number of class names ({len(class_names)}) and number of probability columns ({probs.shape[1]})."
        )

    # Create DataFrame
    submission_df = pd.DataFrame(probs, columns=class_names)

    # Insert the 'id' column at the beginning
    submission_df.insert(0, "id", ids)

    # Ensure probabilities are within [0, 1] as per requirements
    # While models usually output valid probs, we clip to be safe against numerical instability
    numeric_cols = submission_df.columns.drop("id")
    submission_df[numeric_cols] = submission_df[numeric_cols].clip(lower=0.0, upper=1.0)

    # Save to CSV without the index
    submission_df.to_csv(output_path, index=False)
