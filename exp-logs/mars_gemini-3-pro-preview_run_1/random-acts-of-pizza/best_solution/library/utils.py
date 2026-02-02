import os
import random
import numpy as np
import pandas as pd
import torch
from library import config


def set_seed(seed=config.RANDOM_STATE):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(debug=None):
    """
    Loads the train, validation, and test datasets from the paths specified in config.

    Args:
        debug (bool): If True, samples a small subset of the data for debugging.

    Returns:
        tuple: (df_train, df_val, df_test) as pandas DataFrames.
    """
    if debug is None:
        debug = config.DEBUG

    if not os.path.exists(config.TRAIN_DATA_PATH):
        raise FileNotFoundError(f"Train data not found at {config.TRAIN_DATA_PATH}")
    if not os.path.exists(config.VAL_DATA_PATH):
        raise FileNotFoundError(f"Validation data not found at {config.VAL_DATA_PATH}")
    if not os.path.exists(config.TEST_DATA_PATH):
        raise FileNotFoundError(f"Test data not found at {config.TEST_DATA_PATH}")

    df_train = pd.read_csv(config.TRAIN_DATA_PATH)
    df_val = pd.read_csv(config.VAL_DATA_PATH)
    df_test = pd.read_csv(config.TEST_DATA_PATH)

    if debug:
        # Sample a subset for debugging
        sample_size = min(config.DEBUG_SAMPLE_SIZE, len(df_train))
        df_train = df_train.sample(
            n=sample_size, random_state=config.RANDOM_STATE
        ).reset_index(drop=True)

        sample_size_val = min(config.DEBUG_SAMPLE_SIZE, len(df_val))
        df_val = df_val.sample(
            n=sample_size_val, random_state=config.RANDOM_STATE
        ).reset_index(drop=True)

        sample_size_test = min(config.DEBUG_SAMPLE_SIZE, len(df_test))
        df_test = df_test.sample(
            n=sample_size_test, random_state=config.RANDOM_STATE
        ).reset_index(drop=True)

    return df_train, df_val, df_test


def get_common_columns(df_train, df_test, exclude_cols=None):
    """
    Identifies the intersection of columns between train and test sets to prevent leakage,
    excluding specific identifier or target columns.

    Args:
        df_train (pd.DataFrame): Training dataframe.
        df_test (pd.DataFrame): Test dataframe.
        exclude_cols (list, optional): Additional columns to exclude.

    Returns:
        list: A sorted list of common column names to be used as features.
    """
    # Default columns to exclude (identifiers, targets, leakage)
    default_exclusions = {
        "requester_received_pizza",
        "request_id",
        "source_file",
        "giver_username_if_known",
        "request_text",
        "request_title",
        "request_text_edit_aware",
        "requester_subreddits_at_request",  # Usually a list/string, handled separately in feature engineering
        "requester_username",
    }

    if exclude_cols:
        default_exclusions.update(exclude_cols)

    # Find intersection
    train_cols = set(df_train.columns)
    test_cols = set(df_test.columns)
    common_cols = train_cols.intersection(test_cols)

    # Remove exclusions
    feature_cols = [col for col in common_cols if col not in default_exclusions]

    # Sort for deterministic order
    return sorted(feature_cols)


def save_submission(request_ids, probabilities, output_path=config.SUBMISSION_PATH):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        request_ids (array-like): List or array of request IDs.
        probabilities (array-like): List or array of predicted probabilities.
        output_path (str): Path to save the submission file.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission_df = pd.DataFrame(
        {"request_id": request_ids, "requester_received_pizza": probabilities}
    )

    submission_df.to_csv(output_path, index=False)
