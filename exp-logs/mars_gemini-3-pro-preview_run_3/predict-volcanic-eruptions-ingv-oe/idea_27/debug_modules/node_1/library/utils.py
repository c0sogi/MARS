import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from library import config


def load_metadata(split, debug=config.DEBUG):
    """
    Loads the metadata for a specific data split.

    Args:
        split (str): The split to load. Must be one of 'train', 'val', or 'test'.
        debug (bool): If True, samples a subset of the data for debugging purposes.
                      Defaults to config.DEBUG.

    Returns:
        pd.DataFrame: The metadata DataFrame containing segment_ids and file_paths.
                      For 'train' and 'val', also contains 'time_to_eruption'.
    """
    if split == "train":
        path = config.TRAIN_METADATA_PATH
    elif split == "val":
        path = config.VAL_METADATA_PATH
    elif split == "test":
        path = config.TEST_METADATA_PATH
    else:
        raise ValueError(
            f"Invalid split '{split}'. Expected 'train', 'val', or 'test'."
        )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at: {path}")

    df = pd.read_csv(path)

    if debug:
        # Sample the dataset for debugging
        sample_size = min(config.DEBUG_SAMPLE_SIZE, len(df))
        df = df.sample(n=sample_size, random_state=config.SEED).reset_index(drop=True)
        print(f"DEBUG MODE: Loaded {len(df)} samples from {split} metadata.")

    return df


def read_sensor_file(file_path):
    """
    Reads a raw sensor data CSV file.

    Args:
        file_path (str): The relative file path from the input directory
                         (e.g., 'train/1000015382.csv').

    Returns:
        pd.DataFrame: The sensor data loaded as float32.
    """
    full_path = os.path.join(config.INPUT_DIR, file_path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Sensor file not found: {full_path}")

    # Load as float32 to handle potential NaNs and adhere to dataset notes
    return pd.read_csv(full_path, dtype="float32")


def calculate_mae(y_true, y_pred):
    """
    Calculates the Mean Absolute Error (MAE) between true and predicted values.

    Args:
        y_true (array-like): Ground truth target values.
        y_pred (array-like): Predicted target values.

    Returns:
        float: The MAE score.
    """
    mae = mean_absolute_error(y_true, y_pred)
    # Print full precision without formatting
    print(f"Validation MAE: {mae}")
    return mae


def save_submission(predictions, test_meta_df):
    """
    Formats and saves the predictions to a CSV file for submission.

    Args:
        predictions (array-like): The predicted time_to_eruption values.
        test_meta_df (pd.DataFrame): The test metadata DataFrame containing 'segment_id'.
                                     Must correspond row-wise to predictions.
    """
    if len(predictions) != len(test_meta_df):
        raise ValueError(
            f"Length mismatch: predictions ({len(predictions)}) vs test metadata ({len(test_meta_df)})"
        )

    submission = pd.DataFrame(
        {"segment_id": test_meta_df["segment_id"], "time_to_eruption": predictions}
    )

    # Ensure the directory exists
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)

    submission.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
