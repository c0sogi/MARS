import os
import random
import numpy as np
import pandas as pd
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def laplace_log_likelihood(true_fvc, pred_fvc, pred_sigma):
    """
    Calculates the modified Laplace Log Likelihood metric as defined in the task.

    The metric is defined as:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        true_fvc (np.array or float): Ground truth FVC values.
        pred_fvc (np.array or float): Predicted FVC values.
        pred_sigma (np.array or float): Predicted confidence (sigma) values.

    Returns:
        float: The mean metric score over the input data. Higher (less negative) is better.
    """
    # Ensure inputs are numpy arrays for vectorized computation
    true_fvc = np.array(true_fvc, dtype=np.float64)
    pred_fvc = np.array(pred_fvc, dtype=np.float64)
    pred_sigma = np.array(pred_sigma, dtype=np.float64)

    # Clip the confidence values to a minimum of 70 ml
    sigma_clipped = np.maximum(pred_sigma, 70)

    # Calculate absolute error and clip it to a maximum of 1000 ml
    delta = np.minimum(np.abs(true_fvc - pred_fvc), 1000)

    # Calculate the metric
    sqrt_2 = np.sqrt(2)
    metric_values = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    # Return the mean score
    return np.mean(metric_values)


def create_submission(patient_weeks, fvc_preds, confidence_preds, output_path=None):
    """
    Creates a submission CSV file in the required format.

    Args:
        patient_weeks (list or np.array): List of 'Patient_Week' identifiers.
        fvc_preds (list or np.array): Predicted FVC values.
        confidence_preds (list or np.array): Predicted Confidence values.
        output_path (str, optional): Destination path for the CSV.
                                     Defaults to Config.SUBMISSION_PATH.
    """
    if output_path is None:
        output_path = Config.SUBMISSION_PATH

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame
    submission_df = pd.DataFrame(
        {
            "Patient_Week": patient_weeks,
            "FVC": fvc_preds,
            "Confidence": confidence_preds,
        }
    )

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
