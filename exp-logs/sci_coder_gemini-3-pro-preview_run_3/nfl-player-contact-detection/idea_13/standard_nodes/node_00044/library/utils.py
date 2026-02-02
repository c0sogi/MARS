import os
import random
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from library.config import SEED


def seed_everything(seed=SEED):
    """
    Sets the random seed for Python, NumPy, and environment variables
    to ensure reproducible results.

    Args:
        seed (int): The seed value to use. Defaults to the value in config.py.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    # Note: Torch or other library seeds can be added here if needed in the future,
    # but strictly following the prompt's scope for now.


def compute_mcc(y_true, y_pred):
    """
    Calculates the Matthews Correlation Coefficient (MCC) between true labels
    and predicted labels.

    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets as returned by a classifier.

    Returns:
        float: The MCC score.
    """
    # Ensure inputs are numpy arrays for consistency
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    score = matthews_corrcoef(y_true, y_pred)
    return score


def get_dataframe_hash(df):
    """
    Generates a SHA-256 hash based on the content of a pandas DataFrame.
    Useful for cache invalidation strategies.

    Args:
        df (pd.DataFrame): The dataframe to hash.

    Returns:
        str: A hexadecimal hash string representing the dataframe's content.
    """
    if df is None:
        return ""

    # hash_pandas_object returns a series of hash values (one per row)
    # We include the index to ensure order matters
    row_hashes = pd.util.hash_pandas_object(df, index=True).values

    # Hash the byte representation of the array of row hashes
    full_hash = hashlib.sha256(row_hashes.tobytes()).hexdigest()

    return full_hash
