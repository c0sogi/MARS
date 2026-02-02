import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_auc(y_true, y_pred):
    """
    Computes the Area Under the ROC Curve (AUC).

    Args:
        y_true: True binary labels.
        y_pred: Predicted probabilities.

    Returns:
        float: The ROC AUC score.
    """
    return roc_auc_score(y_true, y_pred)


def ensure_directory(path: str):
    """
    Ensures that the directory for the given file path exists.
    """
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)


def save_submission(submission_df: pd.DataFrame, path: str):
    """
    Safely saves the submission DataFrame to a CSV file.

    Args:
        submission_df: The DataFrame containing predictions.
        path: The file path to save the CSV.
    """
    ensure_directory(path)
    submission_df.to_csv(path, index=False)
    print(f"Submission saved to {path}")
