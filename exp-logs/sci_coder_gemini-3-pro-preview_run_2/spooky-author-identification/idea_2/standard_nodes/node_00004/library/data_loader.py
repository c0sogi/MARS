import os
import pandas as pd
from library.config import Config


def load_data(debug=False, n_debug_samples=100):
    """
    Loads the training, validation, and test datasets from the metadata files defined in Config.

    Args:
        debug (bool): If True, loads only a subset of the data for debugging purposes.
        n_debug_samples (int): Number of samples to load in debug mode.

    Returns:
        tuple: A tuple containing three pandas DataFrames: (train_df, val_df, test_df).
    """
    # Verify file existence
    if not os.path.exists(Config.TRAIN_DATA_PATH):
        raise FileNotFoundError(
            f"Training metadata file not found at: {Config.TRAIN_DATA_PATH}"
        )
    if not os.path.exists(Config.VAL_DATA_PATH):
        raise FileNotFoundError(
            f"Validation metadata file not found at: {Config.VAL_DATA_PATH}"
        )
    if not os.path.exists(Config.TEST_DATA_PATH):
        raise FileNotFoundError(
            f"Test metadata file not found at: {Config.TEST_DATA_PATH}"
        )

    # Load datasets
    train_df = pd.read_csv(Config.TRAIN_DATA_PATH)
    val_df = pd.read_csv(Config.VAL_DATA_PATH)
    test_df = pd.read_csv(Config.TEST_DATA_PATH)

    # Ensure text columns are strings (handling potential NaNs or mixed types if any)
    # Although metadata generation ensures clean CSVs, this is a safety measure.
    if "text" in train_df.columns:
        train_df["text"] = train_df["text"].astype(str)
    if "text" in val_df.columns:
        val_df["text"] = val_df["text"].astype(str)
    if "text" in test_df.columns:
        test_df["text"] = test_df["text"].astype(str)

    # Handle Debug Mode
    if debug:
        print(f"Debug mode enabled: Limiting datasets to {n_debug_samples} samples.")
        train_df = train_df.head(n_debug_samples)
        val_df = val_df.head(n_debug_samples)
        test_df = test_df.head(n_debug_samples)

    return train_df, val_df, test_df
