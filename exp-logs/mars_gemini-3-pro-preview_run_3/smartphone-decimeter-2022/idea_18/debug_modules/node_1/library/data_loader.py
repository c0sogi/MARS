import os
import pandas as pd
import numpy as np
from library.coordinate_utils import WGS84_to_ECEF, ECEF_to_ENU, ECEF_to_WGS84

# Constants
CACHE_DIR = "./working/idea_18"
INPUT_ROOT = "./input"


def load_metadata(split="train"):
    """
    Loads the metadata CSV for the specified split.

    Args:
        split (str): 'train', 'val', or 'test'

    Returns:
        pd.DataFrame: Metadata dataframe containing drive_id, phone_name, and file paths.
    """
    path = os.path.join("./metadata", f"{split}_metadata.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")
    return pd.read_csv(path)


def clean_gnss(gnss_df):
    """
    Filters GNSS data based on signal quality and validity.

    Args:
        gnss_df (pd.DataFrame): Raw GNSS dataframe.

    Returns:
        pd.DataFrame: Filtered GNSS dataframe.
    """
    # Filter Raw messages only
    if "MessageType" in gnss_df.columns:
        gnss_df = gnss_df[gnss_df["MessageType"] == "Raw"].copy()

    # Filter based on BiasUncertainty (sanity check for clock validity)
    if "BiasUncertaintyNanos" in gnss_df.columns:
        gnss_df = gnss_df[gnss_df["BiasUncertaintyNanos"] < 1e9]

    # Filter weak signals (Cn0DbHz >= 20 is a standard threshold for usability)
    if "Cn0DbHz" in gnss_df.columns:
        gnss_df = gnss_df[gnss_df["Cn0DbHz"] >= 20]

    # Filter high uncertainty measurements
    if "ReceivedSvTimeUncertaintyNanos" in gnss_df.columns:
        gnss_df = gnss_df[gnss_df["ReceivedSvTimeUncertaintyNanos"] < 500]

    return gnss_df


def compute_targets(merged_df):
    """
    Computes ENU residuals (GT - WLS) for the merged dataframe.
    Adds 'target_E', 'target_N', 'target_U' columns representing the error in meters.

    Args:
        merged_df (pd.DataFrame): DataFrame containing both GNSS WLS positions and Ground Truth LLA.

    Returns:
        pd.DataFrame: DataFrame with added target columns.
    """
    # Required columns
    wls_cols = [
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]
    gt_cols = ["LatitudeDegrees", "LongitudeDegrees", "AltitudeMeters"]

    if not all(c in merged_df.columns for c in wls_cols + gt_cols):
        return merged_df

    # 1. Extract WLS ECEF
    wls_x = merged_df["WlsPositionXEcefMeters"].values
    wls_y = merged_df["WlsPositionYEcefMeters"].values
    wls_z = merged_df["WlsPositionZEcefMeters"].values

    # Handle NaNs in WLS
    mask_wls = np.isnan(wls_x) | np.isnan(wls_y) | np.isnan(wls_z)

    # Use dummy values for conversion where WLS is NaN to avoid crashing ECEF_to_WGS84
    # These rows will be masked out later
    wls_x_safe = np.where(mask_wls, 6378137.0, wls_x)
    wls_y_safe = np.where(mask_wls, 0.0, wls_y)
    wls_z_safe = np.where(mask_wls, 0.0, wls_z)

    # Convert WLS ECEF -> WLS LLA (Reference for ENU)
    ref_lat, ref_lon, ref_alt = ECEF_to_WGS84(wls_x_safe, wls_y_safe, wls_z_safe)

    # 2. Extract GT LLA
    gt_lat = merged_df["LatitudeDegrees"].values
    gt_lon = merged_df["LongitudeDegrees"].values
    gt_alt_raw = merged_df["AltitudeMeters"].values

    # Handle missing GT Altitude: fill with WLS Altitude (ref_alt) to minimize projection error
    # If ref_alt is also NaN (from mask_wls), it doesn't matter as result will be NaN
    gt_alt = np.where(np.isnan(gt_alt_raw), ref_alt, gt_alt_raw)

    # Convert GT LLA -> GT ECEF
    gt_x, gt_y, gt_z = WGS84_to_ECEF(gt_lat, gt_lon, gt_alt)

    # 3. Convert GT ECEF -> ENU (relative to WLS)
    e, n, u = ECEF_to_ENU(gt_x, gt_y, gt_z, ref_lat, ref_lon, ref_alt)

    # Apply mask for invalid WLS rows
    e[mask_wls] = np.nan
    n[mask_wls] = np.nan
    u[mask_wls] = np.nan

    merged_df["target_E"] = e
    merged_df["target_N"] = n
    merged_df["target_U"] = u

    return merged_df


def load_drive_data(
    drive_id,
    phone_name,
    gnss_rel_path,
    gt_rel_path=None,
    imu_rel_path=None,
    load_cached_data=True,
):
    """
    Loads, cleans, and aligns GNSS and Ground Truth data for a specific drive.

    Args:
        drive_id (str): Drive ID
        phone_name (str): Phone name
        gnss_rel_path (str): Relative path to GNSS file
        gt_rel_path (str, optional): Relative path to GT file
        imu_rel_path (str, optional): Relative path to IMU file
        load_cached_data (bool): Whether to use caching

    Returns:
        pd.DataFrame: Aligned DataFrame with GNSS features and (optional) GT targets.
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    cache_file = os.path.join(CACHE_DIR, f"{drive_id}_{phone_name}.parquet")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_file):
        try:
            df = pd.read_parquet(cache_file)
            return df
        except Exception as e:
            print(
                f"Error loading cache for {drive_id} {phone_name}: {e}. Recomputing..."
            )

    # 2. Compute from scratch

    # Load GNSS
    gnss_path = os.path.join(INPUT_ROOT, gnss_rel_path)
    if not os.path.exists(gnss_path):
        raise FileNotFoundError(f"GNSS file not found: {gnss_path}")

    gnss_df = pd.read_csv(gnss_path)

    # Basic Cleaning
    gnss_df = clean_gnss(gnss_df)

    # Load GT if provided
    if gt_rel_path:
        gt_path = os.path.join(INPUT_ROOT, gt_rel_path)
        if os.path.exists(gt_path):
            gt_df = pd.read_csv(gt_path)

            # Select relevant GT columns
            # We need Lat/Lon/Alt for targets, and Speed/Bearing might be useful features/analysis
            gt_cols = [
                "UnixTimeMillis",
                "LatitudeDegrees",
                "LongitudeDegrees",
                "AltitudeMeters",
                "SpeedMps",
                "BearingDegrees",
            ]
            gt_subset = gt_df[gt_cols]

            # Merge GNSS and GT
            # GNSS 'utcTimeMillis' corresponds to GT 'UnixTimeMillis'
            merged_df = pd.merge(
                gnss_df,
                gt_subset,
                left_on="utcTimeMillis",
                right_on="UnixTimeMillis",
                how="inner",
            )

            # Compute Targets (ENU Residuals)
            merged_df = compute_targets(merged_df)

        else:
            print(f"Warning: GT path provided but file not found: {gt_path}")
            merged_df = gnss_df
            merged_df["target_E"] = np.nan
            merged_df["target_N"] = np.nan
            merged_df["target_U"] = np.nan
    else:
        # Test mode or no GT
        merged_df = gnss_df
        # Ensure target columns exist (as NaNs) for consistency
        merged_df["target_E"] = np.nan
        merged_df["target_N"] = np.nan
        merged_df["target_U"] = np.nan

    # Save to cache
    try:
        merged_df.to_parquet(cache_file)
    except Exception as e:
        print(f"Failed to save cache: {e}")

    return merged_df
