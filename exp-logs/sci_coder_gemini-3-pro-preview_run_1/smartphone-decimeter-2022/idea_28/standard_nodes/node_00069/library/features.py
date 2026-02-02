import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import library.config as config
from library.utils import geodetic_to_enu


# ==========================================
# Coordinate Conversion Helpers
# ==========================================
def ecef_to_lla(x, y, z):
    """
    Convert Earth-Centered, Earth-Fixed (ECEF) coordinates to
    Latitude, Longitude, Altitude (WGS84).
    Vectorized implementation.
    """
    # WGS84 ellipsoid constants
    a = 6378137.0
    e = 8.1819190842622e-2

    asq = a**2
    esq = e**2

    b = np.sqrt(asq * (1 - esq))
    bsq = b**2
    ep = np.sqrt((asq - bsq) / bsq)
    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(a * z, b * p)

    lon = np.arctan2(y, x)
    lat = np.arctan2((z + ep**2 * b * np.sin(th) ** 3), (p - esq * a * np.cos(th) ** 3))

    # Calculate altitude (not strictly needed for this task but part of formula)
    N = a / np.sqrt(1 - esq * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N

    # Convert to degrees
    return np.degrees(lat), np.degrees(lon), alt


# ==========================================
# Feature Engineering Helpers
# ==========================================
def assign_azimuth_sector(azimuth_deg):
    """
    Map azimuth (0-360) to sector index (0-3).
    0: NE (0-90), 1: SE (90-180), 2: SW (180-270), 3: NW (270-360)
    """
    # Fill NaNs with 0 to avoid errors, though they should be filtered out
    azimuth_deg = np.nan_to_num(azimuth_deg, nan=0.0)
    return (azimuth_deg // 90).astype(int) % 4


def assign_quality_strata(signal_type, adr_state):
    """
    Map signal to quality strata.
    0: High Quality (L5-like OR Phase Valid)
    1: Standard (Everything else)
    """
    # High fidelity signal types
    high_fi_types = ["GPS_L5", "GAL_E5A", "BDS_B2A", "QZS_J5"]

    # Check if signal type is high fidelity
    is_high_fi = np.isin(signal_type, high_fi_types)

    # Check if Carrier Phase is valid (Bit 0 of AccumulatedDeltaRangeState)
    # 1 = ADR_STATE_VALID
    is_phase_valid = (adr_state & 1) == 1

    # Stratum 0 if either condition is met, else Stratum 1
    is_high_quality = is_high_fi | is_phase_valid
    return np.where(is_high_quality, 0, 1)


def process_drive(drive_id, phone_name, df_meta, input_dir):
    """
    Process a single drive: load GNSS, merge with metadata targets,
    compute features, and return a dataframe.
    """
    # Construct path to GNSS file
    gnss_path = os.path.join(
        input_dir,
        "train" if "LatitudeDegrees" in df_meta.columns else "test",
        drive_id,
        phone_name,
        "device_gnss.csv",
    )

    if not os.path.exists(gnss_path):
        # Fallback for test set structure if needed, though metadata usually handles this
        # Try finding it via the path in metadata if available
        if "gnss_path" in df_meta.columns:
            gnss_path = os.path.join(input_dir, df_meta.iloc[0]["gnss_path"])

    if not os.path.exists(gnss_path):
        print(f"Warning: GNSS file not found: {gnss_path}")
        return None

    # Load GNSS data
    # We only need specific columns to save memory
    use_cols = [
        "utcTimeMillis",
        "SignalType",
        "SvElevationDegrees",
        "SvAzimuthDegrees",
        "Cn0DbHz",
        "AccumulatedDeltaRangeState",
        "RawPseudorangeUncertaintyMeters",
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]

    try:
        df_gnss = pd.read_csv(gnss_path, usecols=lambda c: c in use_cols)
    except ValueError:
        # Fallback if some columns are missing (e.g. older logs)
        df_gnss = pd.read_csv(gnss_path)

    # 1. Temporal Quantization (Align to 1Hz)
    # Round utcTimeMillis to nearest second
    df_gnss["UnixTimeMillis"] = np.round(df_gnss["utcTimeMillis"] / 1000) * 1000
    df_gnss["UnixTimeMillis"] = df_gnss["UnixTimeMillis"].astype(np.int64)

    # 2. Extract WLS Baseline
    # We take the first valid WLS position per timestamp
    wls_cols = [
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]
    if all(c in df_gnss.columns for c in wls_cols):
        df_wls = df_gnss.groupby("UnixTimeMillis")[wls_cols].mean().reset_index()
        # Convert ECEF to LLA
        lat, lon, _ = ecef_to_lla(
            df_wls["WlsPositionXEcefMeters"].values,
            df_wls["WlsPositionYEcefMeters"].values,
            df_wls["WlsPositionZEcefMeters"].values,
        )
        df_wls["WlsLat"] = lat
        df_wls["WlsLon"] = lon
        df_wls = df_wls.drop(columns=wls_cols)
    else:
        # Should not happen based on dataset desc, but handle gracefully
        df_wls = pd.DataFrame({"UnixTimeMillis": df_gnss["UnixTimeMillis"].unique()})
        df_wls["WlsLat"] = 0.0
        df_wls["WlsLon"] = 0.0

    # 3. Filter Invalid Signals
    # Filter out signals with very low signal-to-noise ratio or missing azimuth/elevation
    df_gnss = df_gnss[
        (df_gnss["Cn0DbHz"] > 0)
        & (df_gnss["SvElevationDegrees"].notna())
        & (df_gnss["SvAzimuthDegrees"].notna())
    ].copy()

    # 4. Feature Engineering: Binning
    df_gnss["Sector"] = assign_azimuth_sector(df_gnss["SvAzimuthDegrees"].values)
    df_gnss["Strata"] = assign_quality_strata(
        df_gnss["SignalType"].values, df_gnss["AccumulatedDeltaRangeState"].values
    )

    # 5. Feature Engineering: Aggregation
    # We aggregate by Timestamp, Sector, Strata
    # Features: Cn0DbHz, SvElevationDegrees
    # Stats: mean, std, min, max

    # Create a composite key for pivoting
    # Key format: s{Sector}_q{Strata}
    df_gnss["bin_id"] = (
        "s" + df_gnss["Sector"].astype(str) + "_q" + df_gnss["Strata"].astype(str)
    )

    # Pivot table for directional features
    # We compute stats for both Cn0 and Elevation
    pivot_features = df_gnss.pivot_table(
        index="UnixTimeMillis",
        columns="bin_id",
        values=["Cn0DbHz", "SvElevationDegrees"],
        aggfunc=["mean", "std", "min", "max"],
    )

    # Flatten MultiIndex columns
    # Format: {Stat}_{Feature}_{Bin} -> e.g., mean_Cn0DbHz_s0_q0
    pivot_features.columns = [
        f"{stat}_{feat}_{bin_id}" for stat, feat, bin_id in pivot_features.columns
    ]
    pivot_features = pivot_features.reset_index()

    # 6. Global Context Features
    # SatCount, RawPseudorangeUncertaintyMeters (mean), Azimuth Centroids

    # Calculate Azimuth Centroids (Signal Weighted)
    # Convert Azimuth to radians
    az_rad = np.deg2rad(df_gnss["SvAzimuthDegrees"].values)
    df_gnss["vec_x"] = df_gnss["Cn0DbHz"] * np.cos(az_rad)
    df_gnss["vec_y"] = df_gnss["Cn0DbHz"] * np.sin(az_rad)

    global_agg = (
        df_gnss.groupby("UnixTimeMillis")
        .agg(
            {
                "Cn0DbHz": "count",  # SatCount
                "RawPseudorangeUncertaintyMeters": "mean",
                "vec_x": "mean",
                "vec_y": "mean",
            }
        )
        .reset_index()
    )

    # Compute centroid angle components
    global_agg["global_SinAzCentroid"] = global_agg["vec_y"] / np.sqrt(
        global_agg["vec_x"] ** 2 + global_agg["vec_y"] ** 2 + 1e-6
    )
    global_agg["global_CosAzCentroid"] = global_agg["vec_x"] / np.sqrt(
        global_agg["vec_x"] ** 2 + global_agg["vec_y"] ** 2 + 1e-6
    )

    global_agg = global_agg.rename(
        columns={
            "Cn0DbHz": "global_SatCount",
            "RawPseudorangeUncertaintyMeters": "global_PrUncMean",
        }
    )
    global_agg = global_agg.drop(columns=["vec_x", "vec_y"])

    # 7. Merge Everything
    # Start with metadata (Ground Truth / Target Timestamps)
    # This ensures we only keep rows we care about
    df_meta_copy = df_meta.copy()
    df_meta_copy["UnixTimeMillis"] = (
        np.round(df_meta_copy["UnixTimeMillis"] / 1000) * 1000
    )
    df_meta_copy["UnixTimeMillis"] = df_meta_copy["UnixTimeMillis"].astype(np.int64)

    df_merged = pd.merge(df_meta_copy, df_wls, on="UnixTimeMillis", how="left")
    df_merged = df_merged.dropna(subset=["WlsLat", "WlsLon"])

    # Merge Features
    df_merged = pd.merge(df_merged, pivot_features, on="UnixTimeMillis", how="left")
    df_merged = pd.merge(df_merged, global_agg, on="UnixTimeMillis", how="left")

    # 8. Compute Targets (ENU Offsets)
    # If LatitudeDegrees exists (Train/Val), compute targets
    if "LatitudeDegrees" in df_merged.columns:
        d_east, d_north = geodetic_to_enu(
            df_merged["LatitudeDegrees"].values,
            df_merged["LongitudeDegrees"].values,
            df_merged["WlsLat"].values,
            df_merged["WlsLon"].values,
        )
        df_merged["target_East"] = d_east
        df_merged["target_North"] = d_north

        valid_mask = (
            np.isfinite(df_merged["target_East"])
            & np.isfinite(df_merged["target_North"])
            & (df_merged["target_East"].abs() < 1e6)
            & (df_merged["target_North"].abs() < 1e6)
        )
        df_merged = df_merged[valid_mask]
    else:
        # Inference mode
        df_merged["target_East"] = 0.0
        df_merged["target_North"] = 0.0

    # 9. Fill Missing Values
    # Features might be NaN if a bin was empty or no GNSS data for a timestamp
    # Fill stats with 0
    feat_cols = [
        c for c in df_merged.columns if "_s" in c and "_q" in c
    ]  # Bin features
    global_cols = [
        "global_SatCount",
        "global_PrUncMean",
        "global_SinAzCentroid",
        "global_CosAzCentroid",
    ]

    df_merged[feat_cols] = df_merged[feat_cols].fillna(0)
    df_merged[global_cols] = df_merged[global_cols].fillna(0)

    # Ensure all expected columns exist (in case some bins were never seen in this drive)
    # We need to guarantee the column order for the model
    # Expected columns based on config
    expected_bin_cols = []
    stats = ["mean", "std", "min", "max"]
    base_feats = ["Cn0DbHz", "SvElevationDegrees"]

    for s in range(config.AZIMUTH_SECTORS):
        for q in range(config.QUALITY_STRATA):
            bin_id = f"s{s}_q{q}"
            for feat in base_feats:
                for stat in stats:
                    col_name = f"{stat}_{feat}_{bin_id}"
                    expected_bin_cols.append(col_name)

    # Add missing columns with 0
    for col in expected_bin_cols:
        if col not in df_merged.columns:
            df_merged[col] = 0.0

    # Reorder columns to ensure consistency
    # Identifiers + Global + Binned + Targets + WLS
    final_cols = (
        ["drive_id", "phone_name", "UnixTimeMillis"]
        + global_cols
        + expected_bin_cols
        + ["target_East", "target_North", "WlsLat", "WlsLon"]
    )

    # Filter to only final columns
    df_merged = df_merged[final_cols]

    if df_merged.empty:
        return None

    return df_merged


def generate_dataset(split, load_cached_data=True):
    """
    Generate the dataset for a specific split (train/val/test).
    Handles caching.
    """
    cache_file = os.path.join(config.CACHE_DIR, f"{split}_processed.parquet")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading {split} data from cache: {cache_file}")
        return pd.read_parquet(cache_file)

    # 2. Process from Scratch
    print(f"Generating {split} data from raw files...")

    # Load Metadata
    meta_file = os.path.join(config.METADATA_DIR, f"{split}_metadata.csv")
    if not os.path.exists(meta_file):
        # Fallback for demo/mini runs where metadata might be named differently
        meta_file = os.path.join(config.WORKING_DIR, f"mini_{split}_meta.csv")
        if not os.path.exists(meta_file):
            raise FileNotFoundError(f"Metadata file not found for split {split}")

    df_meta = pd.read_csv(meta_file)

    # Debug Mode: Sample drives
    if config.DEBUG:
        drives = df_meta["drive_id"].unique()
        if len(drives) > config.DEBUG_DRIVE_COUNT:
            selected_drives = drives[: config.DEBUG_DRIVE_COUNT]
            df_meta = df_meta[df_meta["drive_id"].isin(selected_drives)].copy()
            print(f"DEBUG: Sampled {len(selected_drives)} drives.")

    # Process each drive
    processed_dfs = []

    # Group by drive and phone to process sequentially
    # (A drive might have multiple phones, we process them individually)
    groups = df_meta.groupby(["drive_id", "phone_name"])

    for (drive_id, phone_name), group_df in tqdm(groups, desc=f"Processing {split}"):
        df_drive = process_drive(drive_id, phone_name, group_df, config.INPUT_DIR)
        if df_drive is not None:
            processed_dfs.append(df_drive)

    if not processed_dfs:
        raise RuntimeError(f"No data processed for split {split}. Check input paths.")

    full_df = pd.concat(processed_dfs, ignore_index=True)

    # Sort by drive and time to ensure sequence order
    full_df = full_df.sort_values(
        ["drive_id", "phone_name", "UnixTimeMillis"]
    ).reset_index(drop=True)

    # 3. Save Cache
    print(f"Saving {split} data to cache: {cache_file}")
    full_df.to_parquet(cache_file, index=False)

    return full_df
