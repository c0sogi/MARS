import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import mean_absolute_error
from library.config import SEED


def seed_everything(seed=SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mae_score(y_true, y_pred):
    """
    Calculates the Mean Absolute Error between true and predicted values.

    Args:
        y_true (array-like): Ground truth target values.
        y_pred (array-like): Estimated target values.

    Returns:
        float: The MAE score.
    """
    return mean_absolute_error(y_true, y_pred)


def save_submission(segment_ids, predictions, output_path):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        segment_ids (array-like): List or array of segment IDs.
        predictions (array-like): List or array of predicted time_to_eruption values.
        output_path (str): The file path where the CSV should be saved.
    """
    df = pd.DataFrame({"segment_id": segment_ids, "time_to_eruption": predictions})

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV without index
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def print_metric(name, value):
    """
    Prints a metric name and its value with full precision.

    Args:
        name (str): The name of the metric (e.g., "Validation MAE").
        value (float): The value of the metric.
    """
    print(f"{name}: {value}")
