import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import log_loss
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clip_probabilities(probs):
    """
    Clips probabilities to the range [1e-15, 1 - 1e-15] to ensure numerical stability
    and avoid extremes in the log function, as specified in the metric description.

    Args:
        probs (np.ndarray): Array of predicted probabilities.

    Returns:
        np.ndarray: Clipped probabilities.
    """
    epsilon = 1e-15
    return np.clip(probs, epsilon, 1.0 - epsilon)


def calculate_log_loss(y_true, y_pred):
    """
    Calculates the multi-class logarithmic loss.

    Args:
        y_true (array-like): Ground truth labels (indices).
        y_pred (array-like): Predicted probabilities of shape (n_samples, n_classes).

    Returns:
        float: The calculated log loss.
    """
    # We explicitly provide labels to ensure correct calculation even if a batch
    # is missing a specific class. Assumes y_true are indices [0, 1, 2].
    return log_loss(y_true, y_pred, labels=list(range(Config.NUM_CLASSES)))


def save_submission(ids, probs, output_path=Config.SUBMISSION_PATH):
    """
    Saves the predictions to a CSV file in the format required for submission.

    Args:
        ids (list or np.ndarray): Sequence of sample IDs.
        probs (np.ndarray): Predicted probabilities of shape (n_samples, 3).
                            Columns must correspond to Config.CLASS_NAMES ['EAP', 'HPL', 'MWS'].
        output_path (str): Path where the CSV file will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Apply clipping to ensure the submitted file complies with stability requirements
    clipped_probs = clip_probabilities(probs)

    # Create DataFrame with correct column order
    submission_df = pd.DataFrame(clipped_probs, columns=Config.CLASS_NAMES)
    submission_df.insert(0, "id", ids)

    # Save to CSV without the index
    submission_df.to_csv(output_path, index=False)
