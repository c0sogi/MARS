import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error
from library.config import Config


def load_metadata(split="train"):
    """
    Loads the metadata for a specific split (train, val, or test).

    Args:
        split (str): The dataset split to load. Options: "train", "val", "test".

    Returns:
        pd.DataFrame: The metadata dataframe containing IDs, features, and file paths.
    """
    if split == "train":
        path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        path = Config.VAL_METADATA_PATH
    elif split == "test":
        path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(
            f"Invalid split '{split}'. Expected 'train', 'val', or 'test'."
        )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")

    return pd.read_csv(path)


def compute_rmsle(y_true, y_pred):
    """
    Computes the Mean Column-wise Root Mean Squared Logarithmic Error (RMSLE).

    The metric is calculated by taking the RMSLE for each target column separately
    and then averaging the results.

    Formula:
        RMSLE_col = sqrt( mean( (log(1 + y_true) - log(1 + y_pred))^2 ) )
        Metric = mean( [RMSLE_col1, RMSLE_col2, ...] )

    Args:
        y_true (np.ndarray or pd.DataFrame): Ground truth target values.
        y_pred (np.ndarray or pd.DataFrame): Predicted target values.

    Returns:
        float: The mean column-wise RMSLE score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Ensure no negative values are passed to log function.
    # Physical energies in this dataset are non-negative.
    # We clip predictions to 0.
    y_pred = np.maximum(y_pred, 0)

    # Safety check for ground truth as well
    y_true = np.maximum(y_true, 0)

    # Calculate Mean Squared Logarithmic Error for each column separately
    # multioutput='raw_values' returns an array of MSLE values, one for each output
    msle_per_column = mean_squared_log_error(y_true, y_pred, multioutput="raw_values")

    # Take square root to get RMSLE for each column
    rmsle_per_column = np.sqrt(msle_per_column)

    # Return the mean of the column-wise RMSLEs
    return np.mean(rmsle_per_column)


def save_submission(ids, predictions, filename="submission.csv"):
    """
    Formats and saves the predictions to a CSV file in the required submission format.

    Format:
        id,formation_energy_ev_natom,bandgap_energy_ev
        1,0.1234,1.5678
        ...

    Args:
        ids (list or np.array): Sequence of test sample IDs.
        predictions (np.ndarray): 2D array of shape (n_samples, 2) containing predicted values.
                                  Column 0: formation_energy_ev_natom
                                  Column 1: bandgap_energy_ev
        filename (str): Name of the output file. Defaults to "submission.csv".
    """
    # Validate predictions shape
    if predictions.ndim != 2 or predictions.shape[1] != 2:
        raise ValueError(
            f"Predictions must be a 2D array with 2 columns. Got shape {predictions.shape}"
        )

    # Create DataFrame with correct column names from Config
    submission_df = pd.DataFrame(predictions, columns=Config.TARGET_COLS)

    # Insert the 'id' column at the start
    submission_df.insert(0, "id", ids)

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Construct full output path
    output_path = os.path.join(Config.SUBMISSION_DIR, filename)

    # Save to CSV without index
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
