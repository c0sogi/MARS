import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import latlon_to_enu

# ==========================================
# Constants & Helpers
# ==========================================
# WGS84 Ellipsoid Constants for ECEF to LLA conversion
WGS84_A = 6378137.0
WGS84_B = 6356752.3142
WGS84_E_SQ = 6.69437999014 * 0.001


def ecef_to_lla(x, y, z):
    """
    Converts Earth-Centered Earth-Fixed (ECEF) coordinates to Geodetic coordinates
    (Latitude, Longitude, Altitude).
    """
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    r = np.sqrt(x**2 + y**2)
    Esq = (WGS84_A**2 - WGS84_B**2) / (WGS84_B**2)
    F = 54 * (WGS84_B**2) * (z**2)
    G = r**2 + (1 - WGS84_E_SQ) * (z**2) - WGS84_E_SQ * (Esq * r**2)
    C = (WGS84_E_SQ**2) * F * (r**2) / (G**3)
    S = (1 + C + np.sqrt(C**2 + 2 * C)) ** (1 / 3)
    P = F / (3 * (S + 1 / S + 1) ** 2 * (G**2))
    Q = np.sqrt(1 + 2 * (WGS84_E_SQ**2) * P)
    r_0 = -(P * WGS84_E_SQ * r) / (1 + Q) + np.sqrt(
        0.5 * (WGS84_A**2) * (1 + 1 / Q)
        - (P * (1 - WGS84_E_SQ) * (z**2)) / (Q * (1 + Q))
        - 0.5 * P * (r**2)
    )
    U = np.sqrt((r - WGS84_E_SQ * r_0) ** 2 + z**2)
    V = np.sqrt((r - WGS84_E_SQ * r_0) ** 2 + (1 - WGS84_E_SQ) * z**2)
    Z_0 = (WGS84_B**2 * z) / (WGS84_A * V)

    alt = U * (1 - (WGS84_B**2) / (WGS84_A * V))
    lat = np.arctan((z + Esq * Z_0) / r)
    lon = np.arctan2(y, x)

    return np.degrees(lat), np.degrees(lon), alt


def process_gnss_data(gnss_df):
    """
    Aggregates raw GNSS data into features at 1Hz resolution.
    """
    # 1. Temporal Quantization: Align timestamps to nearest second
    gnss_df["UnixTimeMillis"] = np.round(gnss_df["utcTimeMillis"] / 1000.0) * 1000.0
    gnss_df["UnixTimeMillis"] = gnss_df["UnixTimeMillis"].astype(np.int64)

    # 2. Pre-compute trigonometric values for Azimuth
    rad_az = np.radians(gnss_df["SvAzimuthDegrees"].fillna(0))
    gnss_df["sin_az"] = np.sin(rad_az)
    gnss_df["cos_az"] = np.cos(rad_az)

    # 3. Weighted Moments Preparation (Weight by Signal Strength Cn0)
    gnss_df["Cn0DbHz"] = gnss_df["Cn0DbHz"].fillna(0)
    gnss_df["w_sin_az"] = gnss_df["sin_az"] * gnss_df["Cn0DbHz"]
    gnss_df["w_cos_az"] = gnss_df["cos_az"] * gnss_df["Cn0DbHz"]

    # 4. Define Aggregations
    aggs = {
        "Cn0DbHz": ["mean", "std", "min", "max"],
        "SvElevationDegrees": ["mean", "std", "min", "max"],
        "sin_az": ["mean"],
        "cos_az": ["mean"],
        "w_sin_az": ["sum"],
        "w_cos_az": ["sum"],
        "RawPseudorangeUncertaintyMeters": ["mean"],
        "Svid": ["count"],  # Used as Satellite Count
    }

    # 5. Group and Aggregate
    grouped = gnss_df.groupby("UnixTimeMillis")
    features = grouped.agg(aggs)

    # Flatten MultiIndex columns
    features.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in features.columns]

    # 6. Normalize Weighted Moments
    sum_cn0 = grouped["Cn0DbHz"].sum()
    sum_cn0[sum_cn0 == 0] = 1.0  # Avoid division by zero

    features["weighted_sin_az"] = features["w_sin_az_sum"] / sum_cn0
    features["weighted_cos_az"] = features["w_cos_az_sum"] / sum_cn0

    # Drop intermediate sum columns
    features = features.drop(columns=["w_sin_az_sum", "w_cos_az_sum"])

    # Rename count to SatCount
    features = features.rename(columns={"Svid_count": "SatCount"})

    # Fill NaNs resulting from std of single samples
    features = features.fillna(0)

    # 7. Extract WLS Baseline Position
    # We take the mean WLS position for the second (usually constant or very close)
    if "WlsPositionXEcefMeters" in gnss_df.columns:
        wls_aggs = {
            "WlsPositionXEcefMeters": "mean",
            "WlsPositionYEcefMeters": "mean",
            "WlsPositionZEcefMeters": "mean",
        }
        wls_pos = grouped.agg(wls_aggs)
        features = features.join(wls_pos)

        # Convert WLS ECEF to Lat/Lon
        wls_lat, wls_lon, _ = ecef_to_lla(
            features["WlsPositionXEcefMeters"],
            features["WlsPositionYEcefMeters"],
            features["WlsPositionZEcefMeters"],
        )
        features["wls_lat"] = wls_lat
        features["wls_lon"] = wls_lon

        # Drop ECEF columns to save memory
        features = features.drop(
            columns=[
                "WlsPositionXEcefMeters",
                "WlsPositionYEcefMeters",
                "WlsPositionZEcefMeters",
            ]
        )
    else:
        # Fallback (should not happen given dataset desc)
        features["wls_lat"] = 0.0
        features["wls_lon"] = 0.0

    return features.reset_index()


def load_and_process_dataset(metadata_path, split_name, load_cached_data=True):
    """
    Loads raw data based on metadata, processes it, and returns a combined DataFrame.
    Implements caching to ./working/idea_15/cache/
    """
    cache_file = os.path.join(Config.CACHE_DIR, f"{split_name}_processed.parquet")

    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached {split_name} data from {cache_file}...")
        return pd.read_parquet(cache_file)

    print(f"Processing {split_name} data from scratch...")

    df_meta = pd.read_csv(metadata_path)

    # Get unique drives to iterate over
    unique_drives = df_meta[["drive_id", "phone_name", "gnss_path"]].drop_duplicates()

    processed_dfs = []

    for _, row in unique_drives.iterrows():
        drive_id = row["drive_id"]
        phone_name = row["phone_name"]
        gnss_path = os.path.join(Config.INPUT_DIR, row["gnss_path"])

        if not os.path.exists(gnss_path):
            continue

        # Load Raw GNSS
        try:
            gnss_df = pd.read_csv(gnss_path)
        except Exception as e:
            print(f"Error reading {gnss_path}: {e}")
            continue

        # Process Features
        features_df = process_gnss_data(gnss_df)

        # Add identifiers
        features_df["drive_id"] = drive_id
        features_df["phone_name"] = phone_name

        # If Train/Val, merge with Ground Truth to create Targets
        if "LatitudeDegrees" in df_meta.columns:
            # Get GT for this drive
            gt_subset = df_meta[
                (df_meta["drive_id"] == drive_id)
                & (df_meta["phone_name"] == phone_name)
            ][["UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]]

            # Merge features with GT (Inner join keeps only labeled timestamps)
            merged_df = pd.merge(
                features_df, gt_subset, on="UnixTimeMillis", how="inner"
            )

            # Calculate Target Deltas (ENU Meters)
            # Target = GroundTruth - WLS_Baseline
            d_east, d_north = latlon_to_enu(
                merged_df["LatitudeDegrees"].values,
                merged_df["LongitudeDegrees"].values,
                merged_df["wls_lat"].values,
                merged_df["wls_lon"].values,
            )

            merged_df["target_east"] = d_east
            merged_df["target_north"] = d_north

            # Remove absolute coordinates to prevent leakage
            merged_df = merged_df.drop(columns=["LatitudeDegrees", "LongitudeDegrees"])

            processed_dfs.append(merged_df)

        else:
            # Test set: Keep all processed timestamps.
            # Alignment with sample_submission happens at inference/submission time.
            processed_dfs.append(features_df)

    if not processed_dfs:
        print(f"No data processed for {split_name}!")
        return pd.DataFrame()

    combined_df = pd.concat(processed_dfs, ignore_index=True)

    # 2. Save to Cache
    print(f"Saving processed {split_name} data to {cache_file}...")
    combined_df.to_parquet(cache_file, index=False)

    return combined_df


def get_data(load_cached_data=True):
    """
    Main entry point to get processed train, val, and test dataframes.
    """
    train_df = load_and_process_dataset(
        Config.TRAIN_METADATA_PATH, "train", load_cached_data
    )
    val_df = load_and_process_dataset(Config.VAL_METADATA_PATH, "val", load_cached_data)
    test_df = load_and_process_dataset(
        Config.TEST_METADATA_PATH, "test", load_cached_data
    )

    return train_df, val_df, test_df
