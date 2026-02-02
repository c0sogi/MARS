import os
import sys
import random
import logging
import numpy as np
import pandas as pd
import torch


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


def get_logger(name, log_file=None):
    """
    Creates a logger that outputs to stdout and optionally a file.

    Args:
        name (str): The name of the logger.
        log_file (str, optional): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(message)s")

    # Stream Handler (Stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def calculate_weighted_log_loss(y_true_df, y_pred_df):
    """
    Calculates the weighted multi-label logarithmic loss according to competition metrics.

    Args:
        y_true_df (pd.DataFrame): DataFrame containing ground truth.
                                  Must have columns: ['StudyInstanceUID', 'patient_overall', 'C1', ..., 'C7']
        y_pred_df (pd.DataFrame): DataFrame containing predictions.
                                  Must have columns: ['row_id', 'fractured']

    Returns:
        float: The weighted log loss score.
    """
    # Competition weights: Overall is weighted 7x higher than individual vertebrae
    weights = {
        "patient_overall": 7.0,
        "C1": 1.0,
        "C2": 1.0,
        "C3": 1.0,
        "C4": 1.0,
        "C5": 1.0,
        "C6": 1.0,
        "C7": 1.0,
    }

    # Work on copies to avoid side effects
    pred = y_pred_df.copy()

    # Helper to parse row_id into StudyInstanceUID and Label
    def parse_row(row_id):
        if row_id.endswith("patient_overall"):
            return row_id.replace("_patient_overall", ""), "patient_overall"
        else:
            # Assumes format StudyUID_C#
            parts = row_id.rsplit("_", 1)
            return parts[0], parts[1]

    # Parse row_id
    parsed = pred["row_id"].apply(parse_row)
    pred["StudyInstanceUID"] = parsed.apply(lambda x: x[0])
    pred["label"] = parsed.apply(lambda x: x[1])

    # Map weights to each prediction row
    pred["weight"] = pred["label"].map(weights)

    # Melt truth dataframe to long format for merging (StudyUID, Label) -> Target
    label_cols = ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

    # Ensure input truth dataframe has necessary columns
    missing_cols = [c for c in label_cols if c not in y_true_df.columns]
    if missing_cols:
        raise ValueError(f"y_true_df is missing columns: {missing_cols}")

    truth_melted = y_true_df.melt(
        id_vars=["StudyInstanceUID"],
        value_vars=label_cols,
        var_name="label",
        value_name="target",
    )

    # Merge predictions with truth
    merged = pd.merge(pred, truth_melted, on=["StudyInstanceUID", "label"], how="left")

    # Filter out rows where target is missing (e.g., mismatch in studies)
    merged = merged.dropna(subset=["target"])

    if len(merged) == 0:
        return 0.0

    # Clip predictions for numerical stability in log calculation
    eps = 1e-15
    y_pred = np.clip(merged["fractured"].values, eps, 1 - eps)
    y_true = merged["target"].values
    w = merged["weight"].values

    # Calculate Log Loss: L = - [y * log(p) + (1-y) * log(1-p)]
    log_loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # Apply weights
    weighted_log_loss = w * log_loss

    # Metric is the average across all rows
    score = np.mean(weighted_log_loss)

    return score


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
