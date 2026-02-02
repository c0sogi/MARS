import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import RANDOM_SEED


def set_seed(seed: int = RANDOM_SEED) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to RANDOM_SEED from config.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """
    Returns the available PyTorch device (CUDA or CPU).

    Returns:
        torch.device: The device to be used for computation.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def get_common_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame, exclude_cols: list = None
) -> list:
    """
    Identifies the intersection of columns between train and test DataFrames to prevent leakage,
    excluding specified columns.

    Args:
        train_df (pd.DataFrame): Training data.
        test_df (pd.DataFrame): Test data.
        exclude_cols (list, optional): List of columns to exclude from the intersection
                                       (e.g., target variables, IDs).

    Returns:
        list: A sorted list of common column names.
    """
    if exclude_cols is None:
        exclude_cols = []

    # Find intersection
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    common_cols = train_cols.intersection(test_cols)

    # Remove excluded columns
    final_cols = [col for col in common_cols if col not in exclude_cols]

    return sorted(final_cols)


def save_submission(request_ids: list, predictions: list, output_path: str) -> None:
    """
    Saves predictions to a CSV file in the required submission format.

    Format:
        request_id,requester_received_pizza
        t3_i8iy4,0.123
        ...

    Args:
        request_ids (list): List of request identifiers.
        predictions (list): List of predicted probabilities or labels.
        output_path (str): Path to save the submission CSV.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission_df = pd.DataFrame(
        {"request_id": request_ids, "requester_received_pizza": predictions}
    )

    submission_df.to_csv(output_path, index=False)
