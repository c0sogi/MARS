import pandas as pd
import numpy as np
from library.config import process_gnss_data, aggregate_features, get_sector


def calculate_residuals(gnss_df):
    """
    Computes physics-based residuals for GNSS signals.

    This function calculates:
    1. Pseudorange Residuals: The difference between the corrected measured pseudorange
       and the geometric distance from the satellite to the WLS baseline position.
       Common mode error (receiver clock bias) is removed via epoch-wise median.
    2. Doppler Residuals: The difference between the corrected pseudorange rate and
       the projected satellite velocity (range rate). Common mode drift is removed.

    Args:
        gnss_df (pd.DataFrame): Raw GNSS data containing satellite positions,
                                WLS positions, and raw measurements.

    Returns:
        pd.DataFrame: The input dataframe enriched with 'pr_residual' and
                      'doppler_residual' columns.
    """
    # Delegate to the robust implementation in library.config
    return process_gnss_data(gnss_df)


def assign_sectors(gnss_df):
    """
    Bins satellites into azimuthal sectors based on their position relative to the receiver.

    This spatial binning allows the model to learn directional error corrections
    (e.g., "satellites in the North sector have high positive residuals -> move South").

    Args:
        gnss_df (pd.DataFrame): GNSS data containing 'SvAzimuthDegrees'.

    Returns:
        pd.DataFrame: The input dataframe with a new 'sector' column (integer).
    """
    # Ensure we work on a copy to prevent side effects on the original dataframe
    df = gnss_df.copy()

    # Fill missing azimuths with 0 (North) to ensure valid sector assignment
    azimuths = df["SvAzimuthDegrees"].fillna(0)

    # Map azimuth (0-360) to sector index
    df["sector"] = get_sector(azimuths)

    return df


def aggregate_sector_features(gnss_df, imu_df=None):
    """
    Aggregates GNSS residuals and signal quality metrics by sector, and computes
    global dynamic features from IMU data.

    Feature Engineering Steps:
    1. Sector-based Aggregation: For each sector, computes Mean, Std, Max, Min of
       Pseudorange Residuals, Doppler Residuals, Cn0, and Elevation.
    2. Global Aggregation: Computes global statistics across all satellites per epoch.
    3. IMU Integration: If provided, computes acceleration magnitude statistics
       per epoch to capture device dynamics.

    Args:
        gnss_df (pd.DataFrame): GNSS data with residuals (output of calculate_residuals).
        imu_df (pd.DataFrame, optional): Raw IMU data (accelerometer, etc.).

    Returns:
        pd.DataFrame: A feature matrix indexed by 'utcTimeMillis', ready for
                      alignment with ground truth targets.
    """
    # Delegate to the core aggregation logic
    # Note: aggregate_features handles sector assignment internally if not present,
    # but relies on the residuals being present in gnss_df.
    return aggregate_features(gnss_df, imu_df)
