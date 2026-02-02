import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config, setup_reproducibility


def set_seed(seed: int = Config.SEED) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Wraps the configuration's setup_reproducibility function to ensure
    consistent deterministic behavior.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    setup_reproducibility(seed)


def get_device() -> torch.device:
    """
    Returns the appropriate PyTorch device (CUDA or CPU) based on system availability
    and configuration settings.

    Returns:
        torch.device: The device object to be used for model training and inference.
    """
    return torch.device(Config.DEVICE)


def calculate_roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Area Under the Receiver Operating Characteristic Curve (ROC AUC).
    This is the primary metric for the task.

    Args:
        y_true (np.ndarray or list): True binary labels (0 or 1).
        y_pred (np.ndarray or list): Predicted probabilities for the positive class.

    Returns:
        float: The ROC AUC score. Returns 0.5 if only one class is present in y_true
               (e.g., during small-batch debugging) to prevent execution errors.
    """
    # Ensure inputs are numpy arrays for compatibility with sklearn
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Check for single-class edge case (common in small batches or debug runs)
    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(y_true, y_pred)


def save_submission(
    request_ids: list, probabilities: list, output_path: str = Config.SUBMISSION_FILE
) -> None:
    """
    Saves the predictions to a CSV file in the format required for the competition.

    The output file will contain a header and two columns:
    - request_id: The identifier of the request.
    - requester_received_pizza: The predicted probability of success.

    Args:
        request_ids (list): List of request identifiers (strings).
        probabilities (list): List of predicted probabilities (floats).
        output_path (str): Full path where the CSV file should be saved.
                           Defaults to Config.SUBMISSION_FILE.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create the DataFrame
    submission_df = pd.DataFrame(
        {"request_id": request_ids, "requester_received_pizza": probabilities}
    )

    # Save to CSV without the index
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
