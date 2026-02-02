import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import log_loss
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def window_dicom(img: np.ndarray, window_center: int, window_width: int) -> np.ndarray:
    """
    Applies windowing to a DICOM pixel array and normalizes it to [0, 1].

    The formula used is:
    lower = window_center - window_width // 2
    upper = window_center + window_width // 2
    val = clip(img, lower, upper)
    val = (val - lower) / (upper - lower)

    Args:
        img (np.ndarray): The raw DICOM pixel array.
        window_center (int): The center of the window (level).
        window_width (int): The width of the window.

    Returns:
        np.ndarray: The windowed and normalized image array (float32).
    """
    img = img.astype(np.float32)

    lower = window_center - window_width // 2
    upper = window_center + window_width // 2

    img = np.clip(img, lower, upper)

    # Avoid division by zero
    if window_width > 0:
        img = (img - lower) / window_width
    else:
        img = img - lower  # Should not happen with valid window_width

    return img


def get_weighted_log_loss(
    solution_df: pd.DataFrame, submission_df: pd.DataFrame
) -> float:
    """
    Calculates the weighted multi-label logarithmic loss for the competition.

    Weights:
        - patient_overall: 7
        - C1-C7: 1

    The loss is calculated per row and then averaged.
    L_ij = -w_j * [y_ij * log(p_ij) + (1-y_ij) * log(1-p_ij)]

    Args:
        solution_df (pd.DataFrame): DataFrame containing 'row_id' and 'fractured' (true labels).
        submission_df (pd.DataFrame): DataFrame containing 'row_id' and 'fractured' (predicted probabilities).

    Returns:
        float: The weighted logarithmic loss.
    """
    # Define weights
    # patient_overall is weighted 7, others 1
    weights = {
        "C1": 1,
        "C2": 1,
        "C3": 1,
        "C4": 1,
        "C5": 1,
        "C6": 1,
        "C7": 1,
        "patient_overall": 7,
    }

    # Ensure indices match for alignment
    # We create copies to avoid modifying original dataframes
    y_true = solution_df.copy()
    y_pred = submission_df.copy()

    # Merge to ensure we are comparing the same rows
    # Suffixes are used if columns overlap, but we mainly care about 'fractured'
    merged = pd.merge(y_true, y_pred, on="row_id", suffixes=("_true", "_pred"))

    # Extract class type from row_id to assign weights
    # row_id format: StudyInstanceUID_Class
    # We split by '_' and take the last part.
    # Note: patient_overall contains an underscore, so we need to handle that.
    # '1.2.3_C1' -> 'C1'
    # '1.2.3_patient_overall' -> 'patient_overall'

    def get_weight(row_id):
        if row_id.endswith("patient_overall"):
            return weights["patient_overall"]
        else:
            # Extract the last part after the underscore
            # Assuming format [UID]_[C#]
            suffix = row_id.split("_")[-1]
            return weights.get(suffix, 1)  # Default to 1 if not found

    merged["weight"] = merged["row_id"].apply(get_weight)

    # Calculate Log Loss per row
    # Clip predictions to avoid log(0)
    epsilon = 1e-15
    pred = np.clip(merged["fractured_pred"].values, epsilon, 1 - epsilon)
    true = merged["fractured_true"].values
    weight = merged["weight"].values

    # Binary Cross Entropy formula
    # loss = - (y * log(p) + (1-y) * log(1-p))
    loss = -(true * np.log(pred) + (1 - true) * np.log(1 - pred))

    # Apply weights
    weighted_loss = loss * weight

    # The prompt states: "Finally, loss is averaged across all rows."
    # This implies Sum(Weighted_Loss) / N_rows
    # It does NOT imply Sum(Weighted_Loss) / Sum(Weights) based on the text provided.
    # However, to be safe and consistent with standard weighted loss definitions:
    # If the instruction is literally "averaged across all rows", it is mean(weighted_loss).

    final_metric = np.mean(weighted_loss)

    return float(final_metric)
