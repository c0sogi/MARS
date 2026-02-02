import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_log_loss(y_true, y_pred):
    """
    Calculates the multi-class log loss metric as defined in the competition rules.
    Performs row-wise normalization and clipping prior to scoring.

    Args:
        y_true (array-like): Ground truth labels (indices). Shape (n_samples,).
        y_pred (array-like): Predicted probabilities. Shape (n_samples, n_classes).

    Returns:
        float: The calculated log loss.
    """
    # Ensure inputs are numpy arrays
    y_pred = np.array(y_pred)

    # 1. Rescale rows to sum to 1 (normalization)
    # "The submitted probabilities ... are rescaled prior to being scored"
    row_sums = y_pred.sum(axis=1, keepdims=True)
    # Handle edge case where row sum is 0 (though unlikely with valid model outputs)
    row_sums[row_sums == 0] = 1.0
    y_pred_norm = y_pred / row_sums

    # 2. Clip probabilities to avoid log(0) extremes
    # "predicted probabilities are replaced with max(min(p,1-10^-15),10^-15)"
    eps = 1e-15
    y_pred_clipped = np.clip(y_pred_norm, eps, 1 - eps)

    # 3. Calculate Log Loss
    # We use sklearn's log_loss but pass the pre-processed probabilities
    # labels parameter ensures correct handling if y_true is a subset of classes
    labels = np.arange(y_pred.shape[1])
    score = log_loss(y_true, y_pred_clipped, labels=labels)

    return score


def save_submission(ids, predictions, class_names, output_path=Config.SUBMISSION_PATH):
    """
    Formats and saves the predictions to a CSV file.

    Args:
        ids (array-like): List or array of image IDs.
        predictions (array-like): Matrix of predicted probabilities (n_samples, n_classes).
        class_names (list): List of species names corresponding to the columns of predictions.
        output_path (str): File path to save the submission. Defaults to Config.SUBMISSION_PATH.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame
    submission_df = pd.DataFrame(predictions, columns=class_names)

    # Insert the 'id' column at the beginning
    submission_df.insert(0, "id", ids)

    # Save to CSV without index
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
