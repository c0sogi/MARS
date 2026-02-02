import pandas as pd
import numpy as np
from library.utils import load_dataset


def load_datasets(load_cached_data=True, max_samples=None):
    """
    Loads the train, validation, and test datasets, extracting features and labels.

    Args:
        load_cached_data (bool): If True, attempts to load preprocessed data from cache.
        max_samples (int, optional): If set, truncates the datasets to this number of samples
                                     for debugging purposes.

    Returns:
        tuple: A tuple containing three elements:
            - (X_train, y_train): Tuple of text features and binary labels for training.
            - (X_val, y_val): Tuple of text features and binary labels for validation.
            - (X_test, test_df): Tuple of text features for testing and the full test DataFrame
                                 (needed for submission format).
    """
    # Load DataFrames using the library utility which handles cleaning and caching
    train_df = load_dataset(split="train", load_cached_data=load_cached_data)
    val_df = load_dataset(split="validation", load_cached_data=load_cached_data)
    test_df = load_dataset(split="test", load_cached_data=load_cached_data)

    # Apply debugging limit if requested
    if max_samples is not None:
        train_df = train_df.iloc[:max_samples]
        val_df = val_df.iloc[:max_samples]
        test_df = test_df.iloc[:max_samples]

    # Extract Training Data
    # Ensure text is treated as string and labels as integers
    X_train = train_df["Comment_Clean"].astype(str).values
    y_train = train_df["Insult"].astype(int).values

    # Extract Validation Data
    X_val = val_df["Comment_Clean"].astype(str).values
    y_val = val_df["Insult"].astype(int).values

    # Extract Test Data
    # Test set does not contain labels. We return the dataframe to preserve ID/Date info for submission.
    X_test = test_df["Comment_Clean"].astype(str).values

    return (X_train, y_train), (X_val, y_val), (X_test, test_df)
