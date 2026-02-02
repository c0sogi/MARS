import os
import pandas as pd
import numpy as np
from library.config import CACHE_DIR
from library.data_loader import DataLoader
from library.utils import ecef_to_geodetic


def calculate_residuals(gnss_df):
    """
    Computes the difference between raw pseudoranges and distances to the WLS baseline.
    Also estimates and removes the receiver clock bias.

    Args:
        gnss_df (pd.DataFrame): DataFrame containing GNSS measurements.

    Returns:
        pd.DataFrame: DataFrame with 'Residual' and 'DiffResidual' columns added.
    """
    # Calculate Geometric Distance (Range)
    # Ensure necessary columns exist
    req_cols = [
        "SvPositionXEcefMeters",
        "WlsPositionXEcefMeters",
        "SvPositionYEcefMeters",
        "WlsPositionYEcefMeters",
        "SvPositionZEcefMeters",
        "WlsPositionZEcefMeters",
    ]

    if not all(col in gnss_df.columns for col in req_cols):
        raise ValueError(f"Missing columns for range calculation. Required: {req_cols}")

    gnss_df["Range"] = np.sqrt(
        (gnss_df["SvPositionXEcefMeters"] - gnss_df["WlsPositionXEcefMeters"]) ** 2
        + (gnss_df["SvPositionYEcefMeters"] - gnss_df["WlsPositionYEcefMeters"]) ** 2
        + (gnss_df["SvPositionZEcefMeters"] - gnss_df["WlsPositionZEcefMeters"]) ** 2
    )

    # Calculate Corrected Pseudorange
    gnss_df["Pr_corr"] = (
        gnss_df["RawPseudorangeMeters"]
        + gnss_df["SvClockBiasMeters"].fillna(0)
        - gnss_df["IsrbMeters"].fillna(0)
        - gnss_df["IonosphericDelayMeters"].fillna(0)
        - gnss_df["TroposphericDelayMeters"].fillna(0)
    )

    # Calculate Residual
    gnss_df["Residual"] = gnss_df["Pr_corr"] - gnss_df["Range"]

    # Remove Common Mode Error (Receiver Clock Bias estimate)
    if "UnixTimeMillis" in gnss_df.columns:
        epoch_bias = (
            gnss_df.groupby("UnixTimeMillis")["Residual"].median().reset_index()
        )
        epoch_bias.columns = ["UnixTimeMillis", "ClockBias"]
        gnss_df = pd.merge(gnss_df, epoch_bias, on="UnixTimeMillis", how="left")
        gnss_df["DiffResidual"] = gnss_df["Residual"] - gnss_df["ClockBias"]
    else:
        # Fallback if no time column (treat as single epoch)
        gnss_df["DiffResidual"] = gnss_df["Residual"] - gnss_df["Residual"].median()

    return gnss_df


def project_errors(gnss_df):
    """
    Calculates Line-of-Sight (LOS) vectors and projects residuals onto the East/North plane.
    Requires WLS positions in ECEF. Adds Wls_Lat/Lon if missing.

    Args:
        gnss_df (pd.DataFrame): DataFrame with GNSS data and residuals.

    Returns:
        pd.DataFrame: DataFrame with projected force and covariance columns.
    """
    # Ensure WLS Geodetic coords exist for rotation
    if "Wls_Lat" not in gnss_df.columns:
        lats, lons, _ = ecef_to_geodetic(
            gnss_df["WlsPositionXEcefMeters"].values,
            gnss_df["WlsPositionYEcefMeters"].values,
            gnss_df["WlsPositionZEcefMeters"].values,
        )
        gnss_df["Wls_Lat"] = lats
        gnss_df["Wls_Lon"] = lons

    # Calculate Line-of-Sight (LOS) Unit Vectors in ECEF
    dx = gnss_df["SvPositionXEcefMeters"] - gnss_df["WlsPositionXEcefMeters"]
    dy = gnss_df["SvPositionYEcefMeters"] - gnss_df["WlsPositionYEcefMeters"]
    dz = gnss_df["SvPositionZEcefMeters"] - gnss_df["WlsPositionZEcefMeters"]
    dist = np.sqrt(dx**2 + dy**2 + dz**2)

    ux_ecef = dx / dist
    uy_ecef = dy / dist
    uz_ecef = dz / dist

    # Rotate LOS vectors to local ENU frame
    lat_rad = np.radians(gnss_df["Wls_Lat"].values)
    lon_rad = np.radians(gnss_df["Wls_Lon"].values)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    # Rotation matrix application (ECEF -> ENU)
    u_e = -sin_lon * ux_ecef + cos_lon * uy_ecef
    u_n = -sin_lat * cos_lon * ux_ecef - sin_lat * sin_lon * uy_ecef + cos_lat * uz_ecef

    # Weighting by Signal Strength
    w = gnss_df["Cn0DbHz"].fillna(20)

    # Compute Projected Residual Forces
    if "DiffResidual" not in gnss_df.columns:
        # Fallback if calculate_residuals wasn't called
        gnss_df = calculate_residuals(gnss_df)

    gnss_df["Force_E"] = w * gnss_df["DiffResidual"] * u_e
    gnss_df["Force_N"] = w * gnss_df["DiffResidual"] * u_n

    # Compute Geometry Covariance terms
    gnss_df["Cov_E"] = w * (u_e**2)
    gnss_df["Cov_N"] = w * (u_n**2)
    gnss_df["Cov_EN"] = w * (u_e * u_n)

    return gnss_df


def aggregate_epoch_features(gnss_df):
    """
    Sums projected forces and computes geometry matrix elements per timestamp.

    Args:
        gnss_df (pd.DataFrame): DataFrame with projected errors.

    Returns:
        pd.DataFrame: Aggregated features per epoch.
    """
    agg_funcs = {
        "Force_E": "sum",
        "Force_N": "sum",
        "Cov_E": "sum",
        "Cov_N": "sum",
        "Cov_EN": "sum",
        "Cn0DbHz": "mean",
        "Svid": "count",
    }

    # Keep WLS positions if needed for later
    first_cols = [
        "Wls_Lat",
        "Wls_Lon",
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]
    for col in first_cols:
        if col in gnss_df.columns:
            agg_funcs[col] = "first"

    epoch_feats = gnss_df.groupby("UnixTimeMillis").agg(agg_funcs).reset_index()

    # Rename to match config features
    epoch_feats = epoch_feats.rename(
        columns={
            "Force_E": "NetForce_E",
            "Force_N": "NetForce_N",
            "Cn0DbHz": "Cn0DbHz_mean",
            "Svid": "Svid_count",
        }
    )
    return epoch_feats


def get_train_data(load_cached_data=True):
    """
    Loads the training dataset using the DataLoader.
    Ensures caching logic is followed.
    """
    loader = DataLoader()
    return loader.load_dataset("train", load_cached_data=load_cached_data)


def get_val_data(load_cached_data=True):
    """
    Loads the validation dataset using the DataLoader.
    Ensures caching logic is followed.
    """
    loader = DataLoader()
    return loader.load_dataset("val", load_cached_data=load_cached_data)


def get_test_data(load_cached_data=True):
    """
    Loads the test dataset using the DataLoader.
    Ensures caching logic is followed.
    """
    loader = DataLoader()
    return loader.load_dataset("test", load_cached_data=load_cached_data)
