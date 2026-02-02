import os
import time
import random
import numpy as np
import pandas as pd
import torch
from contextlib import contextmanager
from sklearn.metrics import roc_auc_score
from library.config import SEED, SUBMISSION_PATH


def set_seed(seed: int = SEED) -> None:
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


@contextmanager
def timer(name: str):
    """
    Context manager to measure and print the execution time of a code block.

    Args:
        name (str): The name of the operation being timed.
    """
    t0 = time.time()
    yield
    t1 = time.time()
    print(f"[{name}] done in {t1 - t0:.4f} s")


def calculate_auc(
    y_true: np.ndarray, y_pred: np.ndarray, label: str = "Validation"
) -> float:
    """
    Calculates and prints the Area Under the ROC Curve with full precision.

    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_pred (np.ndarray): Predicted probabilities.
        label (str): Label for the log message.

    Returns:
        float: The calculated AUC score.
    """
    score = roc_auc_score(y_true, y_pred)
    # Printing without formatting to ensure full precision is shown
    print(f"{label} AUC: {score}")
    return score


def save_submission(
    request_ids: np.ndarray, y_pred: np.ndarray, path: str = SUBMISSION_PATH
) -> None:
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        request_ids (np.ndarray): Array of request IDs.
        y_pred (np.ndarray): Array of predicted probabilities.
        path (str): File path to save the submission. Defaults to SUBMISSION_PATH from config.
    """
    # Ensure directory exists
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    df = pd.DataFrame({"request_id": request_ids, "requester_received_pizza": y_pred})

    df.to_csv(path, index=False)
    print(f"Submission saved to {path}")


def load_metadata(path: str) -> pd.DataFrame:
    """
    Helper to load metadata parquet files.

    Args:
        path (str): Path to the parquet file.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")
    return pd.read_parquet(path)
