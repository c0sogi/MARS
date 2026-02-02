import os
import sys
import random
import logging
import numpy as np
import pandas as pd
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str = "RSNA_Fracture"):
    """
    Creates and configures a logger for console output.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers if the logger is retrieved multiple times
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def calculate_weighted_loss(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    """
    Calculates the weighted multi-label logarithmic loss as specified in the competition metric.

    The metric is defined as the average of the weighted binary log loss across all rows.
    There are 8 rows per study: 'patient_overall' and 'C1' through 'C7'.

    Weights:
        - patient_overall: 1.0
        - C1-C7: 1/7 each

    Args:
        y_true (pd.DataFrame): Ground truth DataFrame containing 'StudyInstanceUID'
                               and target columns (patient_overall, C1..C7).
        y_pred (pd.DataFrame): Predictions DataFrame containing 'StudyInstanceUID'
                               and predicted probabilities for the target columns.

    Returns:
        float: The calculated weighted log loss.
    """
    # Define weights based on task description:
    # "The any label is weighted more highly than specific fracture level sub-types."
    # Standard competition weights: Overall=1, Each Vertebra=1/7.
    weights = {
        "patient_overall": 1.0,
        "C1": 1.0 / 7.0,
        "C2": 1.0 / 7.0,
        "C3": 1.0 / 7.0,
        "C4": 1.0 / 7.0,
        "C5": 1.0 / 7.0,
        "C6": 1.0 / 7.0,
        "C7": 1.0 / 7.0,
    }

    target_cols = list(weights.keys())

    # Validation: Ensure required columns exist
    for col in ["StudyInstanceUID"] + target_cols:
        if col not in y_true.columns:
            raise ValueError(f"Column '{col}' missing from y_true DataFrame")
        if col not in y_pred.columns:
            raise ValueError(f"Column '{col}' missing from y_pred DataFrame")

    # Align DataFrames by StudyInstanceUID
    # Using inner join to ensure we only evaluate on common studies
    merged = pd.merge(
        y_true[["StudyInstanceUID"] + target_cols],
        y_pred[["StudyInstanceUID"] + target_cols],
        on="StudyInstanceUID",
        suffixes=("_true", "_pred"),
    )

    if len(merged) == 0:
        return 0.0

    total_weighted_loss = 0.0
    epsilon = 1e-15

    # Calculate weighted loss for each column
    for col, w in weights.items():
        y_t = merged[f"{col}_true"].values.astype(np.float64)
        y_p = merged[f"{col}_pred"].values.astype(np.float64)

        # Clip predictions to avoid log(0)
        y_p = np.clip(y_p, epsilon, 1.0 - epsilon)

        # Binary Cross Entropy: -[y * log(p) + (1-y) * log(1-p)]
        bce = -(y_t * np.log(y_p) + (1.0 - y_t) * np.log(1.0 - y_p))

        # Apply weight to the loss
        # L_ij = -w_j * [...]
        weighted_bce = w * bce

        # Sum up the weighted losses
        total_weighted_loss += np.sum(weighted_bce)

    # The metric specifies: "Finally, loss is averaged across all rows."
    # Total rows = Number of studies * Number of labels (8)
    n_studies = len(merged)
    n_labels = len(target_cols)
    total_rows = n_studies * n_labels

    final_metric = total_weighted_loss / total_rows

    return final_metric
