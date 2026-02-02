import os
import random
import numpy as np
import pandas as pd
import torch
from library.config import Config


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


def compute_log_loss(y_true, y_pred):
    """
    Computes the multi-class logarithmic loss with specific clipping and rescaling
    as defined in the task metric description.

    Steps:
    1. Rescale each row of probabilities to sum to 1.
    2. Clip probabilities to the range [1e-15, 1 - 1e-15].
    3. Compute the negative log likelihood of the true classes.

    Args:
        y_true: Array-like of shape (n_samples,) containing true class labels (strings or ints),
                OR (n_samples, n_classes) one-hot encoded.
        y_pred: Array-like of shape (n_samples, n_classes) containing predicted probabilities.
                Columns must correspond to [EAP, HPL, MWS] (indices 0, 1, 2).

    Returns:
        float: The calculated log loss.
    """
    y_pred = np.array(y_pred, dtype=np.float64)
    y_true = np.array(y_true)

    # 1. Rescale rows to sum to 1 (Metric Requirement)
    row_sums = y_pred.sum(axis=1)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1.0
    y_pred = y_pred / row_sums[:, np.newaxis]

    # 2. Clip probabilities (Metric Requirement)
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # 3. Prepare Ground Truth Indices
    # Handle One-Hot Encoding
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true_indices = np.argmax(y_true, axis=1)
    # Handle String Labels
    elif y_true.dtype.type is np.str_ or isinstance(y_true.flat[0], str):
        y_true_indices = np.array([Config.LABEL2ID[label] for label in y_true])
    # Handle Integer Labels
    else:
        y_true_indices = y_true.astype(int)

    # 4. Calculate Log Loss
    n_samples = len(y_true_indices)
    # Select the probability assigned to the true class for each sample
    true_class_probs = y_pred[np.arange(n_samples), y_true_indices]

    score = -np.mean(np.log(true_class_probs))

    return score


def save_submission(ids, probabilities, output_path=None):
    """
    Formats and saves the submission CSV file.

    Args:
        ids: List or array of sentence IDs.
        probabilities: (N, 3) array of predicted probabilities.
                       Order must correspond to [EAP, HPL, MWS].
        output_path: File path to save the CSV. Defaults to Config.SUBMISSION_FILE_PATH.
    """
    if output_path is None:
        output_path = Config.SUBMISSION_FILE_PATH

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Define columns based on task requirement: id,EAP,HPL,MWS
    cols = ["EAP", "HPL", "MWS"]

    submission_df = pd.DataFrame(probabilities, columns=cols)
    submission_df.insert(0, "id", ids)

    submission_df.to_csv(output_path, index=False)
