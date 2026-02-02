import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from library.config import SG_WINDOW, SG_POLY


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing values in the sensor data with the mean of each column.

    This strategy is chosen to preserve the DC offset (baseline) of the
    seismic signals, which contains valuable information about the
    sensor's state and environment.

    Args:
        df (pd.DataFrame): Raw sensor data containing potential NaNs.

    Returns:
        pd.DataFrame: Dataframe with missing values imputed.
    """
    # Calculate column means, ignoring NaNs
    means = df.mean()

    # Fill NaNs with the calculated means
    # In the rare case a column is all NaNs, mean is NaN; fill those with 0.0
    df_imputed = df.fillna(means).fillna(0.0)

    return df_imputed


def apply_savitzky_golay(
    df: pd.DataFrame, window_size: int = SG_WINDOW, poly_order: int = SG_POLY
) -> pd.DataFrame:
    """
    Applies a Savitzky-Golay filter to the sensor data to generate a smoothed
    signal stream (Stream B).

    This smoothed stream is essential for calculating robust kinematic features
    (velocity, acceleration) without amplifying high-frequency noise.

    Args:
        df (pd.DataFrame): The input dataframe (typically Stream A).
        window_size (int): The length of the filter window (must be odd).
        poly_order (int): The order of the polynomial used to fit the samples.

    Returns:
        pd.DataFrame: A new dataframe containing the smoothed sensor readings.
    """
    # Apply the filter along the time axis (axis 0)
    # df.values provides the underlying numpy array of shape (n_samples, n_sensors)
    smoothed_values = savgol_filter(
        df.values, window_length=window_size, polyorder=poly_order, axis=0
    )

    return pd.DataFrame(smoothed_values, columns=df.columns, index=df.index)


def preprocess_segment(file_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads a raw sensor data file and prepares the dual-stream representation
    required for the Hybrid-Stream feature engineering pipeline.

    Args:
        file_path (str): The path to the CSV file containing sensor logs.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - Stream A: The raw data with missing values imputed (Intensity View).
            - Stream B: The smoothed data (Kinematic View).
    """
    # Load data as float32 to handle potential NaNs and optimize memory usage
    # The dataset description explicitly notes the need for float32 due to nulls
    df = pd.read_csv(file_path, dtype="float32")

    # 1. Generate Stream A: Raw Intensity with Imputation
    stream_a = impute_missing_values(df)

    # 2. Generate Stream B: Smoothed Signal for Kinematics
    stream_b = apply_savitzky_golay(stream_a)

    return stream_a, stream_b
