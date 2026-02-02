import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import wgs84_to_local_meters

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------


def ecef_to_lla(x, y, z):
    """
    Convert Earth-Centered, Earth-Fixed (ECEF) coordinates to
    Latitude, Longitude, Altitude (LLA) using WGS84 constants.
    """
    # WGS84 ellipsoid constants
    a = 6378137.0
    b = 6356752.31424518
    e = np.sqrt(1 - (b / a) ** 2)
    ep = np.sqrt((a**2 - b**2) / b**2)

    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(a * z, b * p)

    lon = np.arctan2(y, x)
    lat = np.arctan2(
        (z + ep**2 * b * np.sin(th) ** 3), (p - e**2 * a * np.cos(th) ** 3)
    )

    # Calculate altitude (N is the radius of curvature in the prime vertical)
    sin_lat = np.sin(lat)
    N = a / np.sqrt(1 - e**2 * sin_lat**2)
    alt = p / np.cos(lat) - N

    # Convert to degrees
    lat = np.degrees(lat)
    lon = np.degrees(lon)

    return lat, lon, alt


def _process_gnss_data(gnss_path):
    """
    Reads a GNSS CSV file, aggregates signal metrics by epoch,
    and extracts the baseline WLS position.
    """
    full_path = os.path.join(Config.INPUT_DIR, gnss_path)
    if not os.path.exists(full_path):
        # Return empty dataframe with expected columns if file missing
        return pd.DataFrame(
            columns=[
                "UnixTimeMillis",
                "SatelliteCount",
                "MeanCn0",
                "MeanUncertainty",
                "WlsLat",
                "WlsLon",
                "WlsAlt",
            ]
        )

    df = pd.read_csv(full_path)

    # Columns to aggregate
    # We aggregate by utcTimeMillis which corresponds to the epoch
    agg_dict = {
        "Svid": "count",
        "Cn0DbHz": "mean",
        "RawPseudorangeUncertaintyMeters": "mean",
        "WlsPositionXEcefMeters": "first",
        "WlsPositionYEcefMeters": "first",
        "WlsPositionZEcefMeters": "first",
    }

    # Filter for columns that actually exist in the file
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}

    if "utcTimeMillis" not in df.columns:
        return pd.DataFrame()

    df_agg = df.groupby("utcTimeMillis").agg(agg_dict).reset_index()

    # Rename columns to match Config.INPUT_FEATURES
    df_agg.rename(
        columns={
            "utcTimeMillis": "UnixTimeMillis",
            "Svid": "SatelliteCount",
            "Cn0DbHz": "MeanCn0",
            "RawPseudorangeUncertaintyMeters": "MeanUncertainty",
        },
        inplace=True,
    )

    # Convert WLS ECEF to LLA
    if "WlsPositionXEcefMeters" in df_agg.columns:
        lat, lon, alt = ecef_to_lla(
            df_agg["WlsPositionXEcefMeters"].values,
            df_agg["WlsPositionYEcefMeters"].values,
            df_agg["WlsPositionZEcefMeters"].values,
        )
        df_agg["WlsLat"] = lat
        df_agg["WlsLon"] = lon
        df_agg["WlsAlt"] = alt

        # Drop ECEF columns
        df_agg.drop(
            columns=[
                "WlsPositionXEcefMeters",
                "WlsPositionYEcefMeters",
                "WlsPositionZEcefMeters",
            ],
            inplace=True,
        )
    else:
        # If WLS columns missing, fill with NaNs
        df_agg["WlsLat"] = np.nan
        df_agg["WlsLon"] = np.nan
        df_agg["WlsAlt"] = np.nan

    return df_agg


# -------------------------------------------------------------------------
# Main Data Loading Function
# -------------------------------------------------------------------------


def load_dataset(mode="train", load_cached_data=True):
    """
    Loads and preprocesses the dataset for the given mode.

    Args:
        mode: 'train', 'val', or 'test'.
        load_cached_data: If True, attempts to load processed parquet from cache.

    Returns:
        pd.DataFrame: Processed dataframe with features and targets (if applicable).
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"{mode}_data.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"[{mode}] Loading cached data from {cache_path}...")
        return pd.read_parquet(cache_path)

    print(f"[{mode}] Computing data from scratch...")

    # 2. Load Metadata
    if mode == "train":
        meta_path = Config.TRAIN_METADATA_PATH
    elif mode == "val":
        meta_path = Config.VAL_METADATA_PATH
    else:
        meta_path = Config.TEST_METADATA_PATH

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df_meta = pd.read_csv(meta_path)

    # 3. Process Trips
    processed_trips = []
    unique_trips = df_meta["tripId"].unique()

    for i, trip_id in enumerate(unique_trips):
        # Extract metadata for this trip
        trip_meta = df_meta[df_meta["tripId"] == trip_id].copy()
        trip_meta = trip_meta.sort_values("UnixTimeMillis").reset_index(drop=True)

        # Get GNSS path (assume first row has valid path)
        gnss_path = trip_meta["gnss_path"].iloc[0]

        # Process GNSS raw data
        df_gnss = _process_gnss_data(gnss_path)

        # Merge Metadata with GNSS features
        # We use LEFT join to keep all metadata timestamps (required for submission)
        # GNSS data might be missing for some seconds
        df_merged = pd.merge(trip_meta, df_gnss, on="UnixTimeMillis", how="left")

        # 4. Handle Missing Data (Interpolation)
        # Critical for WLS baseline continuity
        cols_to_interpolate = [
            "WlsLat",
            "WlsLon",
            "WlsAlt",
            "SatelliteCount",
            "MeanCn0",
            "MeanUncertainty",
        ]
        # Ensure columns exist
        for col in cols_to_interpolate:
            if col not in df_merged.columns:
                df_merged[col] = np.nan

        # Interpolate linearly based on time
        df_merged[cols_to_interpolate] = df_merged[cols_to_interpolate].interpolate(
            method="linear", limit_direction="both"
        )

        # Fill remaining NaNs (e.g. if whole trip missing GNSS) with 0 or reasonable defaults
        df_merged["SatelliteCount"] = df_merged["SatelliteCount"].fillna(0)
        df_merged["MeanCn0"] = df_merged["MeanCn0"].fillna(0)
        df_merged["MeanUncertainty"] = df_merged["MeanUncertainty"].fillna(
            100
        )  # High uncertainty default

        # If WLS is still missing (very rare), we can't do much.
        # For training we drop, for test we might forward fill from other trips (unlikely needed).
        if mode != "test":
            df_merged = df_merged.dropna(subset=["WlsLat", "WlsLon"])
        else:
            # For test, we must preserve rows. Fill with 0 (will result in bad pred but keeps structure)
            df_merged["WlsLat"] = df_merged["WlsLat"].fillna(0)
            df_merged["WlsLon"] = df_merged["WlsLon"].fillna(0)
            df_merged["WlsAlt"] = df_merged["WlsAlt"].fillna(0)

        # 5. Feature Engineering: Deltas (Velocity)
        # Calculate diffs
        df_merged["DeltaLat"] = df_merged["WlsLat"].diff().fillna(0)
        df_merged["DeltaLon"] = df_merged["WlsLon"].diff().fillna(0)
        df_merged["DeltaAlt"] = df_merged["WlsAlt"].diff().fillna(0)

        # Drop rows with any NaNs in input features to prevent training crashes (Cite debug_lesson_2)
        # Moved after feature engineering to ensure Delta columns exist (Cite debug_lesson_4)
        if mode != "test":
            df_merged = df_merged.dropna(subset=Config.INPUT_FEATURES)

        # 6. Calculate Targets (Train/Val only)
        if mode in ["train", "val"]:
            d_east, d_north = wgs84_to_local_meters(
                df_merged["WlsLat"].values,
                df_merged["WlsLon"].values,
                df_merged["LatitudeDegrees"].values,
                df_merged["LongitudeDegrees"].values,
            )
            df_merged["DeltaEast"] = d_east
            df_merged["DeltaNorth"] = d_north

        processed_trips.append(df_merged)

    # Concatenate all trips
    final_df = pd.concat(processed_trips, ignore_index=True)

    # 7. Select Columns
    cols_to_keep = (
        ["tripId", "UnixTimeMillis"] + Config.INPUT_FEATURES + Config.BASELINE_COLS
    )
    if mode in ["train", "val"]:
        cols_to_keep += Config.TARGET_COLUMNS
        cols_to_keep += ["LatitudeDegrees", "LongitudeDegrees"]

    # Ensure all columns exist
    for col in cols_to_keep:
        if col not in final_df.columns:
            final_df[col] = 0

    final_df = final_df[cols_to_keep]

    # 8. Save to Cache
    print(f"[{mode}] Saving processed data to {cache_path}...")
    final_df.to_parquet(cache_path, index=False)

    return final_df


# -------------------------------------------------------------------------
# Scaler Management
# -------------------------------------------------------------------------


def fit_scaler(train_df):
    """
    Fits a StandardScaler on the training data features and saves statistics to JSON.
    """
    scaler = StandardScaler()
    scaler.fit(train_df[Config.INPUT_FEATURES])

    stats = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "features": Config.INPUT_FEATURES,
    }

    stats_path = os.path.join(Config.CACHE_DIR, "scaler_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f)

    print(f"Scaler statistics saved to {stats_path}")
    return scaler


def transform_data(df):
    """
    Loads scaler statistics and transforms the input features of the dataframe.
    """
    stats_path = os.path.join(Config.CACHE_DIR, "scaler_stats.json")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            "Scaler stats not found. Run fit_scaler on training data first."
        )

    with open(stats_path, "r") as f:
        stats = json.load(f)

    # Verify features match
    if stats["features"] != Config.INPUT_FEATURES:
        raise ValueError("Scaler features do not match current configuration.")

    scaler = StandardScaler()
    scaler.mean_ = np.array(stats["mean"])
    scaler.scale_ = np.array(stats["scale"])

    # Sklearn requires these to be set for transform to work without fitting
    scaler.var_ = scaler.scale_**2
    scaler.n_samples_seen_ = 0

    df_scaled = df.copy()
    df_scaled[Config.INPUT_FEATURES] = scaler.transform(df[Config.INPUT_FEATURES])

    return df_scaled
