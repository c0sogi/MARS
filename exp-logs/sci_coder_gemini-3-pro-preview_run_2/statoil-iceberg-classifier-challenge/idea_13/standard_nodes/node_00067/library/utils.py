import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import log_loss
from library.config import SEED


def seed_everything(seed: int = SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Returns the appropriate PyTorch device (CUDA or CPU).

    Returns:
        torch.device: The selected device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_score(y_true, y_pred):
    """
    Calculates the Log Loss metric for binary classification.

    Args:
        y_true: Array-like of ground truth labels (0 or 1).
        y_pred: Array-like of predicted probabilities for class 1.

    Returns:
        float: The calculated log loss.
    """
    # Ensure inputs are numpy arrays for consistency
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return log_loss(y_true, y_pred)


def print_metrics(metrics):
    """
    Prints metric values with full precision.

    Args:
        metrics (dict): Dictionary of metric names and values.
    """
    for key, value in metrics.items():
        print(f"{key}: {value}")


def save_submission(ids, predictions, output_path):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        ids: List or array of image IDs.
        predictions: List or array of predicted probabilities for 'is_iceberg'.
        output_path: Path to save the CSV file.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.DataFrame({"id": ids, "is_iceberg": predictions})

    # Save to CSV without index
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
