import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from library.config import SAVGOL_WINDOW, SAVGOL_POLYORDER


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing values in the DataFrame with the column-wise mean.
    If a column is entirely NaN, it is filled with 0.

    Args:
        df (pd.DataFrame): The input dataframe containing sensor data.

    Returns:
        pd.DataFrame: The dataframe with missing values imputed.
    """
    # Calculate column means (ignoring NaNs by default)
    means = df.mean()

    # Fill NaNs with the calculated means
    df_filled = df.fillna(means)

    # If any columns were entirely NaN, the mean would be NaN, so fill those with 0
    df_filled = df_filled.fillna(0)

    return df_filled


def apply_savgol_filter(
    df: pd.DataFrame, window_length: int = None, polyorder: int = None
) -> pd.DataFrame:
    """
    Applies a Savitzky-Golay filter to smooth sensor readings to suppress noise.

    Args:
        df (pd.DataFrame): The input dataframe containing sensor data.
        window_length (int, optional): The length of the filter window.
                                       Defaults to SAVGOL_WINDOW from config.
        polyorder (int, optional): The order of the polynomial used to fit the samples.
                                   Defaults to SAVGOL_POLYORDER from config.

    Returns:
        pd.DataFrame: The smoothed dataframe with the same structure as input.
    """
    if window_length is None:
        window_length = SAVGOL_WINDOW
    if polyorder is None:
        polyorder = SAVGOL_POLYORDER

    # Apply the filter along axis 0 (rows/time)
    # savgol_filter returns a numpy array
    smoothed_values = savgol_filter(
        df, window_length=window_length, polyorder=polyorder, axis=0
    )

    # Reconstruct the DataFrame to preserve column names and indices
    df_smoothed = pd.DataFrame(smoothed_values, columns=df.columns, index=df.index)

    return df_smoothed
