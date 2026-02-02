import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from library.config import SUBMISSION_PATH, ID_COL, TARGET_COL


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_submission(ids, probabilities, filename=SUBMISSION_PATH):
    """
    Formats and saves the prediction results to a CSV file.

    Args:
        ids (array-like): List of request_ids.
        probabilities (array-like): List of predicted probabilities for the positive class.
        filename (str): Path to save the submission CSV. Defaults to SUBMISSION_PATH.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Create the DataFrame
    submission_df = pd.DataFrame({ID_COL: ids, TARGET_COL: probabilities})

    # Save to CSV without index
    submission_df.to_csv(filename, index=False)
    print(f"Submission saved to {filename}")


def print_metrics(y_true, y_pred, set_name="Validation"):
    """
    Calculates and prints the ROC AUC score with full precision.

    Args:
        y_true (array-like): True binary labels.
        y_pred (array-like): Predicted probabilities.
        set_name (str): Name of the dataset being evaluated (e.g., 'Validation').
    """
    auc = roc_auc_score(y_true, y_pred)
    # Print full precision as requested
    print(f"{set_name} ROC AUC: {auc}")
