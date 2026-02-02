import os
import random
import numpy as np
import pandas as pd
import torch
from library.config import METADATA_DIR, SUBMISSION_DIR, SEED


def seed_everything(seed: int = SEED):
    """
    Sets the random seed for various libraries to ensure reproducibility.

    Args:
        seed (int): The random seed to use. Defaults to SEED from config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_metadata(split: str) -> pd.DataFrame:
    """
    Loads the metadata CSV file for a specific data split.

    Args:
        split (str): The name of the split ('train', 'val', or 'test').

    Returns:
        pd.DataFrame: The loaded metadata containing segment_ids and file paths.

    Raises:
        ValueError: If an invalid split name is provided.
        FileNotFoundError: If the metadata file does not exist.
    """
    valid_splits = {"train", "val", "test"}
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Expected one of {valid_splits}")

    file_path = os.path.join(METADATA_DIR, f"{split}.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Metadata file for split '{split}' not found at {file_path}"
        )

    return pd.read_csv(file_path)


def save_submission(
    predictions: np.ndarray, test_df: pd.DataFrame, save_path: str = None
):
    """
    Formats and saves the predictions to a CSV file in the required submission format.

    Args:
        predictions (np.ndarray): Array of predicted time_to_eruption values.
        test_df (pd.DataFrame): The test metadata DataFrame containing 'segment_id'.
        save_path (str, optional): The full path where the submission CSV will be saved.
                                   If None, defaults to 'submission.csv' in SUBMISSION_DIR.
    """
    if save_path is None:
        save_path = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Create submission DataFrame
    submission = pd.DataFrame()
    submission["segment_id"] = test_df["segment_id"].astype(int)
    submission["time_to_eruption"] = predictions

    # Save to CSV
    submission.to_csv(save_path, index=False)
    print(f"Submission saved successfully to {save_path}")
