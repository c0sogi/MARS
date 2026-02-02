import os
import pandas as pd
import numpy as np
from library.config import Config, load_and_process_data


def load_raw_data(file_path):
    """
    Loads a dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)


def clean_text(text):
    """
    Preprocesses request narratives.
    Handles NaNs and strips whitespace.
    Can be applied to 'request_text_edit_aware' or 'request_title'.

    Args:
        text (str or float): Input text.

    Returns:
        str: Cleaned text.
    """
    if pd.isna(text):
        return ""
    return str(text).strip()


def get_data_splits():
    """
    Loads the raw train, validation, and test datasets from the metadata directory.
    Uses the paths defined in Config to ensure consistency.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    train_df = load_raw_data(Config.TRAIN_PATH)
    val_df = load_raw_data(Config.VAL_PATH)
    test_df = load_raw_data(Config.TEST_PATH)

    # Optional: Apply basic cleaning to text columns if raw access is needed
    # Note: The deep learning pipeline uses pre-processed features from get_processed_features
    text_cols = ["request_text_edit_aware", "request_title"]
    for df in [train_df, val_df, test_df]:
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].apply(clean_text)

    return train_df, val_df, test_df


def get_processed_features(load_cached_data=True):
    """
    Retrieves the fully processed feature sets for the Hybrid Ensemble (RF + MLP).

    This function delegates to `library.config.load_and_process_data`, which handles:
    - SBERT Embedding generation for Title, Body, and User History.
    - TF-IDF vectorization.
    - Computation of Interaction Features and Consistency Scalars.
    - Caching of the processed numpy arrays to disk.

    Args:
        load_cached_data (bool): If True, attempts to load from the cache directory
                                 defined in Config. If False or cache missing,
                                 re-computes features.

    Returns:
        dict: A dictionary containing the processed datasets:
            - 'rf_train', 'rf_val', 'rf_test': Features for Random Forest.
            - 'mlp_train', 'mlp_val', 'mlp_test': Feature dicts for MLP.
            - 'y_train', 'y_val': Target labels.
            - 'ids_test': Request IDs for submission.
    """
    # The library function implements the caching logic and feature engineering
    # described in Idea 46.
    return load_and_process_data(load_cached_data=load_cached_data)
