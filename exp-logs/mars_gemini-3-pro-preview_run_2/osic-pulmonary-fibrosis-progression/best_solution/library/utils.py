import numpy as np
import pandas as pd
import os


def clip_uncertainty(sigma):
    """
    Clips the uncertainty (confidence) values at a minimum of 70 ml.

    Args:
        sigma (np.ndarray or float): The predicted confidence values (standard deviation).

    Returns:
        np.ndarray or float: The clipped confidence values.
    """
    return np.maximum(sigma, 70)


def calculate_metric(y_true, y_pred, sigma):
    """
    Calculates the modified Laplace Log Likelihood metric.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (np.ndarray): True FVC values.
        y_pred (np.ndarray): Predicted FVC values.
        sigma (np.ndarray): Predicted confidence (uncertainty) values.

    Returns:
        float: The average metric score over the input samples.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sigma = np.array(sigma)

    sigma_clipped = clip_uncertainty(sigma)
    delta = np.minimum(np.abs(y_true - y_pred), 1000)

    metric = -(np.sqrt(2) * delta) / sigma_clipped - np.log(np.sqrt(2) * sigma_clipped)

    return np.mean(metric)


def format_submission(test_df, predictions, confidences, output_path):
    """
    Formats the predictions into the required submission CSV format.

    Args:
        test_df (pd.DataFrame): DataFrame containing the 'Patient_Week' column corresponding to the predictions.
        predictions (np.ndarray): Predicted FVC values.
        confidences (np.ndarray): Predicted confidence values.
        output_path (str): Path to save the submission CSV.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission = pd.DataFrame(
        {
            "Patient_Week": test_df["Patient_Week"],
            "FVC": predictions,
            "Confidence": confidences,
        }
    )

    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
