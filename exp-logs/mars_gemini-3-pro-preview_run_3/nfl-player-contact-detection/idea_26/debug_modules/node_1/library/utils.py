import os
import random
import numpy as np
import torch
import pandas as pd
import hashlib
from sklearn.metrics import matthews_corrcoef
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for various libraries to ensure reproducibility.

    Args:
        seed (int): The random seed value. Defaults to Config.SEED.
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


def get_dataframe_hash(df: pd.DataFrame) -> str:
    """
    Generates a unique MD5 hash for a pandas DataFrame based on its content.
    Useful for cache invalidation strategies.

    Args:
        df (pd.DataFrame): The dataframe to hash.

    Returns:
        str: The MD5 hash string.
    """
    # Hash the values using pandas utility
    # We sort index to ensure order doesn't affect hash if content is same but shuffled
    # However, for time-series/step data, order might matter.
    # Given the context of tracking data, we assume the dataframe passed is in a deterministic state.

    # Using hash_pandas_object is faster than serializing to string
    obj_hash = pd.util.hash_pandas_object(df, index=True).values

    # Create MD5 hash of the underlying numpy array bytes
    hasher = hashlib.md5()
    hasher.update(obj_hash.tobytes())

    # Also incorporate column names to catch schema changes
    col_names = pd.Series(df.columns).astype(str)
    col_hash = pd.util.hash_pandas_object(col_names, index=False).values
    hasher.update(col_hash.tobytes())

    return hasher.hexdigest()


def compute_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Matthews Correlation Coefficient.

    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_pred (np.ndarray): Predicted binary labels.

    Returns:
        float: The MCC score.
    """
    return matthews_corrcoef(y_true, y_pred)


def optimize_threshold(
    y_true: np.ndarray, y_pred_proba: np.ndarray, num_steps: int = 100
):
    """
    Performs a linear search to find the probability threshold that maximizes MCC.

    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_pred_proba (np.ndarray): Predicted probabilities (between 0 and 1).
        num_steps (int): Number of threshold steps to check between 0 and 1.

    Returns:
        tuple: (best_threshold, best_mcc)
    """
    best_mcc = -1.0
    best_threshold = 0.5

    # Generate thresholds to test. We avoid 0.0 and 1.0 strictly to avoid empty classes if possible
    thresholds = np.linspace(0.01, 0.99, num_steps)

    for thresh in thresholds:
        y_pred_binary = (y_pred_proba >= thresh).astype(int)

        # Calculate MCC
        # Note: matthews_corrcoef handles cases with 0 variance gracefully (returns 0.0)
        # but we wrap it just in case of unexpected errors
        try:
            score = matthews_corrcoef(y_true, y_pred_binary)
        except ValueError:
            score = 0.0

        if score > best_mcc:
            best_mcc = score
            best_threshold = thresh

    return best_threshold, best_mcc
