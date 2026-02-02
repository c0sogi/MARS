import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed: int = Config.RANDOM_SEED) -> None:
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.RANDOM_SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Torch seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(
    debug: bool = Config.DEBUG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads the train, validation, and test datasets from the metadata Parquet files.

    Args:
        debug (bool): If True, loads only a subset of the data for debugging purposes.
                      Defaults to Config.DEBUG.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing (train_df, val_df, test_df).
    """
    # Load data from Parquet files
    train_df = pd.read_parquet(Config.TRAIN_PATH)
    val_df = pd.read_parquet(Config.VAL_PATH)
    test_df = pd.read_parquet(Config.TEST_PATH)

    # Apply debug sampling if requested
    if debug:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    return train_df, val_df, test_df


def get_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Area Under the ROC Curve (ROC-AUC).

    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_pred (np.ndarray): Predicted probabilities.

    Returns:
        float: The ROC-AUC score.
    """
    return roc_auc_score(y_true, y_pred)


def save_submission(
    request_ids: list | np.ndarray, predictions: list | np.ndarray
) -> None:
    """
    Saves the predictions to a CSV file in the format required for submission.

    Args:
        request_ids (list | np.ndarray): The list of request IDs.
        predictions (list | np.ndarray): The list of predicted probabilities.
    """
    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Create submission DataFrame
    submission_df = pd.DataFrame(
        {"request_id": request_ids, "requester_received_pizza": predictions}
    )

    # Save to CSV without index
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
