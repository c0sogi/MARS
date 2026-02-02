import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across various libraries.

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


class AverageMeter:
    """
    Computes and stores the average and current value.
    """

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


def metric_laplace_log_likelihood(y_true, y_pred, sigma):
    """
    Calculates the modified Laplace Log Likelihood metric.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|y_true - y_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (np.array): Ground truth FVC values.
        y_pred (np.array): Predicted FVC values.
        sigma (np.array): Predicted confidence (std dev).

    Returns:
        float: The average metric score (higher is better, values are negative).
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sigma = np.array(sigma)

    # Clipping confidence
    sigma_clipped = np.maximum(sigma, Config.METRIC_CLIP_SIGMA)

    # Calculating delta with error thresholding
    delta = np.abs(y_true - y_pred)
    delta = np.minimum(delta, Config.METRIC_MAX_ERROR)

    # Calculating the metric
    sqrt_2 = np.sqrt(2)
    metric = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    return np.mean(metric)


def save_submission_file(submission_df, output_path=None):
    """
    Formats and saves the submission DataFrame.

    Args:
        submission_df (pd.DataFrame): DataFrame containing 'Patient_Week', 'FVC', and 'Confidence'.
        output_path (str, optional): Path to save the CSV. Defaults to Config.SUBMISSION_PATH.
    """
    if output_path is None:
        output_path = Config.SUBMISSION_PATH

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Ensure required columns exist
    required_columns = ["Patient_Week", "FVC", "Confidence"]
    if not all(col in submission_df.columns for col in required_columns):
        raise ValueError(
            f"Submission DataFrame must contain columns: {required_columns}"
        )

    # Select and order columns
    final_df = submission_df[required_columns]

    # Save to CSV
    final_df.to_csv(output_path, index=False)
    # print(f"Submission saved to {output_path}")
