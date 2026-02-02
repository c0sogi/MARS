import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility by delegating to the Config class.
    """
    Config.set_seed(seed)


def calculate_log_loss(y_true, y_pred):
    """
    Calculates the multi-class logarithmic loss with specific normalization and clipping.

    The metric rescales each row of probabilities to sum to 1 and clips them
    to the range [1e-15, 1-1e-15] before scoring.

    Args:
        y_true: Array-like of true labels. Can be strings ('EAP', 'HPL', 'MWS')
                or integers (0, 1, 2).
        y_pred: Array-like of predicted probabilities (shape: [n_samples, 3]).
                The columns must correspond to the classes ['EAP', 'HPL', 'MWS'].
                Can be a numpy array or torch.Tensor.

    Returns:
        float: The calculated log loss.
    """
    # Convert torch tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Normalize probabilities so that each row sums to 1
    # This matches the competition metric requirement: "rescaled prior to being scored"
    row_sums = y_pred.sum(axis=1)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1.0
    y_pred_norm = y_pred / row_sums[:, np.newaxis]

    # Determine labels based on y_true type
    # The columns of y_pred are assumed to be EAP, HPL, MWS in that order.
    if np.issubdtype(y_true.dtype, np.number):
        # If y_true is numeric (e.g. encoded integers), we assume 0->EAP, 1->HPL, 2->MWS
        labels = [0, 1, 2]
    else:
        # If y_true contains strings, we specify the labels to ensure correct mapping
        labels = ["EAP", "HPL", "MWS"]

    # Calculate log loss
    # eps=1e-15 corresponds to the clipping rule max(min(p, 1-10^-15), 10^-15)
    score = log_loss(y_true, y_pred_norm, labels=labels, eps=1e-15)

    return score


def save_submission(ids, probs, output_path=Config.SUBMISSION_FILE):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        ids: Array-like of sample IDs.
        probs: Array-like of predicted probabilities (shape: [n_samples, 3]).
               Columns must correspond to ['EAP', 'HPL', 'MWS'].
               Can be a numpy array or torch.Tensor.
        output_path: Path to save the submission file.
    """
    # Convert to numpy if tensor
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()

    # Create DataFrame with correct column order
    submission = pd.DataFrame(probs, columns=["EAP", "HPL", "MWS"])
    submission.insert(0, "id", ids)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    submission.to_csv(output_path, index=False)
