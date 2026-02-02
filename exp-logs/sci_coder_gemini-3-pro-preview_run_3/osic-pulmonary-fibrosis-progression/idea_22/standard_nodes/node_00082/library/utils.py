import os
import numpy as np
import pandas as pd
import torch
from library.config import Config, seed_everything


def calculate_metric(y_true, y_pred, sigma_pred):
    """
    Calculates the modified Laplace Log Likelihood metric as defined in the competition.

    Metric Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        metric = - (sqrt(2) * delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (np.array or torch.Tensor): True FVC values.
        y_pred (np.array or torch.Tensor): Predicted FVC values.
        sigma_pred (np.array or torch.Tensor): Predicted confidence (sigma).

    Returns:
        float: The average metric score (negative value, higher is better).
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(sigma_pred, torch.Tensor):
        sigma_pred = sigma_pred.detach().cpu().numpy()

    # Ensure inputs are flattened
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    sigma_pred = sigma_pred.flatten()

    # Retrieve constants from Config
    sigma_clip_threshold = Config.SIGMA_CLIP
    max_error_threshold = Config.MAX_ERROR
    sqrt2 = Config.SQRT2

    # Apply clipping to sigma (confidence)
    sigma_clipped = np.maximum(sigma_pred, sigma_clip_threshold)

    # Calculate absolute error and clip it
    delta = np.abs(y_true - y_pred)
    delta = np.minimum(delta, max_error_threshold)

    # Compute the metric
    # Term 1: Scaled error
    term1 = (sqrt2 * delta) / sigma_clipped
    # Term 2: Log penalty
    term2 = np.log(sqrt2 * sigma_clipped)

    metric = -term1 - term2

    return np.mean(metric)


def save_submission(predictions_df, output_path):
    """
    Validates and saves the submission DataFrame to a CSV file.

    Args:
        predictions_df (pd.DataFrame): DataFrame containing predictions.
                                       Must have columns: ['Patient_Week', 'FVC', 'Confidence']
        output_path (str): File path to save the CSV.
    """
    required_columns = ["Patient_Week", "FVC", "Confidence"]

    # Validate columns
    if not all(col in predictions_df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in predictions_df.columns]
        raise ValueError(f"Submission DataFrame missing required columns: {missing}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Select required columns and save
    submission_data = predictions_df[required_columns]
    submission_data.to_csv(output_path, index=False)
    print(f"Submission file saved successfully to: {output_path}")
