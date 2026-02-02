import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import wgs84_to_cartesian

# WGS84 Ellipsoid Constants
A = 6378137.0
B = 6356752.31424518
E_SQ = 6.69437999014e-3
E_PRIME_SQ = 6.73949674228e-3


def ecef_to_geodetic(x, y, z):
    """
    Convert ECEF coordinates to Geodetic (Latitude, Longitude, Altitude).
    Vectorized implementation using numpy.
    """
    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * A, p * B)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    lon = np.arctan2(y, x)

    lat = np.arctan2(z + E_PRIME_SQ * B * (sin_theta**3), p - E_SQ * A * (cos_theta**3))

    # Convert to degrees
    lat_deg = np.degrees(lat)
    lon_deg = np.degrees(lon)

    return lat_deg, lon_deg


def compute_distributional_features(gnss_df):
    """
    Computes distributional embeddings (histograms) and boundary statistics
    for satellite signals at each timestamp.
    """
    # Group by timestamp
    grouped = gnss_df.groupby("utcTimeMillis")

    features_list = []
    timestamps = []

    # Define bin edges
    cn0_bins = np.linspace(
        Config.CN0_RANGE[0], Config.CN0_RANGE[1], Config.CN0_BINS + 1
    )
    elev_bins = np.linspace(
        Config.ELEVATION_RANGE[0], Config.ELEVATION_RANGE[1], Config.ELEVATION_BINS + 1
    )

    for time_millis, group in grouped:
        timestamps.append(time_millis)

        # Extract signals
        cn0 = group["Cn0DbHz"].values
        elev = group["SvElevationDegrees"].values

        # Handle NaNs
        cn0 = cn0[~np.isnan(cn0)]
        elev = elev[~np.isnan(elev)]

        # 1. Histograms
        cn0_hist, _ = np.histogram(cn0, bins=cn0_bins)
        elev_hist, _ = np.histogram(elev, bins=elev_bins)

        # Normalize histograms (density)
        sat_count = len(cn0)
        if sat_count > 0:
            cn0_hist = cn0_hist / sat_count
            elev_hist = elev_hist / sat_count

        # 2. Boundary Statistics & Aggregations
        if sat_count > 0:
            cn0_min = np.min(cn0)
            cn0_max = np.max(cn0)
            cn0_mean = np.mean(cn0)

            elev_min = np.min(elev)
            elev_max = np.max(elev)
            elev_mean = np.mean(elev)
        else:
            cn0_min = cn0_max = cn0_mean = 0.0
            elev_min = elev_max = elev_mean = 0.0

        # 3. Uncertainty (if available)
        if "RawPseudorangeUncertaintyMeters" in group.columns:
            unc = group["RawPseudorangeUncertaintyMeters"].values
            unc = unc[~np.isnan(unc)]
            unc_mean = np.mean(unc) if len(unc) > 0 else 0.0
        else:
            unc_mean = 0.0

        # Concatenate all features
        # Vector size: CN0_BINS + ELEV_BINS + 6 stats + 1 count + 1 uncertainty
        feature_vector = np.concatenate(
            [
                cn0_hist,
                elev_hist,
                [cn0_min, cn0_max, cn0_mean],
                [elev_min, elev_max, elev_mean],
                [sat_count, unc_mean],
            ]
        )

        features_list.append(feature_vector)

    return np.array(features_list), np.array(timestamps)


def prepare_drive_data(drive_id, phone_name, df_meta, load_cached_data=True):
    """
    Loads, processes, and aligns data for a specific drive.
    Returns features, targets, and metadata.
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"{drive_id}_{phone_name}.npz")

    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            return {
                "features": data["features"],
                "targets": data["targets"] if "targets" in data else None,
                "timestamps": data["timestamps"],
                "baseline": data["baseline"],
                "gt_coords": data["gt_coords"] if "gt_coords" in data else None,
            }
        except Exception as e:
            print(f"Failed to load cache for {drive_id}_{phone_name}: {e}")
            # Fall through to recompute

    # 2. Load Raw Data
    # Find the row in metadata to get paths.
    # Note: df_meta can be train, val, or test metadata.
    # We filter by drive_id and phone_name.
    subset = df_meta[
        (df_meta["drive_id"] == drive_id) & (df_meta["phone_name"] == phone_name)
    ]

    if subset.empty:
        # If not in metadata (e.g. might happen during debug if metadata filtered), return None
        return None

    # Get paths from the first row (paths are constant for the drive)
    row = subset.iloc[0]
    gnss_path = os.path.join(Config.INPUT_DIR, row["gnss_path"])

    if not os.path.exists(gnss_path):
        print(f"GNSS file not found: {gnss_path}")
        return None

    gnss_df = pd.read_csv(gnss_path)

    # 3. Feature Engineering
    features, timestamps = compute_distributional_features(gnss_df)

    # 4. Baseline Extraction (WLS from GNSS)
    # We need one baseline position per timestamp.
    # GNSS data has multiple rows per timestamp. We take the first one that has valid WLS data.
    # We group by timestamp and take the first.
    # Note: 'timestamps' array comes from compute_distributional_features which groups by utcTimeMillis.
    # We need to ensure alignment.

    # Efficient way: drop duplicates on timestamp to get one row per epoch
    gnss_unique = gnss_df.drop_duplicates(subset=["utcTimeMillis"]).set_index(
        "utcTimeMillis"
    )

    # Reindex to match the computed features order
    gnss_aligned = gnss_unique.reindex(timestamps)

    # Extract ECEF coordinates
    wls_x = gnss_aligned["WlsPositionXEcefMeters"].values
    wls_y = gnss_aligned["WlsPositionYEcefMeters"].values
    wls_z = gnss_aligned["WlsPositionZEcefMeters"].values

    # Convert to Geodetic (Baseline Lat/Lon)
    # Handle NaNs if any (fill with 0 or forward fill - usually WLS is continuous)
    # Simple forward fill for baseline gaps
    df_wls = pd.DataFrame({"x": wls_x, "y": wls_y, "z": wls_z})
    df_wls = df_wls.ffill().bfill()

    base_lat, base_lon = ecef_to_geodetic(
        df_wls["x"].values, df_wls["y"].values, df_wls["z"].values
    )
    baseline = np.stack([base_lat, base_lon], axis=1)

    # 5. Target Generation (if Ground Truth exists)
    targets = None
    gt_coords = None

    # Check if we have GT columns in metadata (Train/Val)
    if "LatitudeDegrees" in subset.columns:
        # The metadata contains GT for specific timestamps.
        # We need to align the computed features (at GNSS timestamps) with GT timestamps.

        # Convert to Series for easy lookup
        gt_times = subset["UnixTimeMillis"].values
        gt_lats = subset["LatitudeDegrees"].values
        gt_lons = subset["LongitudeDegrees"].values

        # Create arrays for targets aligned with GNSS timestamps (default to NaN)
        aligned_gt_lat = np.full(len(timestamps), np.nan)
        aligned_gt_lon = np.full(len(timestamps), np.nan)

        # For each GT point, find nearest GNSS point within tolerance (e.g. 1000ms)
        # Since both are sorted, we can use merge_asof
        df_gnss_time = pd.DataFrame(
            {"time": timestamps, "idx": np.arange(len(timestamps))}
        )
        df_gt_data = pd.DataFrame({"time": gt_times, "lat": gt_lats, "lon": gt_lons})

        # merge_asof
        aligned = pd.merge_asof(
            df_gnss_time.sort_values("time"),
            df_gt_data.sort_values("time"),
            on="time",
            direction="nearest",
            tolerance=1000,  # 1 second tolerance
        )

        # Fill the arrays
        valid_mask = ~aligned["lat"].isna()
        valid_indices = aligned.loc[valid_mask, "idx"].values.astype(int)
        valid_lats = aligned.loc[valid_mask, "lat"].values
        valid_lons = aligned.loc[valid_mask, "lon"].values

        aligned_gt_lat[valid_indices] = valid_lats
        aligned_gt_lon[valid_indices] = valid_lons

        # Compute Offsets (Targets)
        # Target = (GT - Baseline) converted to Meters
        # Where GT is NaN, Target is NaN

        # Vectorized wgs84_to_cartesian handles arrays
        t_north, t_east = wgs84_to_cartesian(
            aligned_gt_lat, aligned_gt_lon, base_lat, base_lon
        )

        targets = np.stack([t_north, t_east], axis=1)  # Shape (N, 2), contains NaNs
        gt_coords = np.stack([aligned_gt_lat, aligned_gt_lon], axis=1)

    # 6. Save to Cache
    save_dict = {"features": features, "timestamps": timestamps, "baseline": baseline}
    if targets is not None:
        save_dict["targets"] = targets
        save_dict["gt_coords"] = gt_coords

    np.savez_compressed(cache_path, **save_dict)

    return {
        "features": features,
        "targets": targets,
        "timestamps": timestamps,
        "baseline": baseline,
        "gt_coords": gt_coords,
    }


def load_data(split="train", load_cached_data=True):
    """
    High-level function to load all drives for a specific split.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to use cached .npz files.

    Returns:
        list of dicts: Each dict contains data for one drive.
    """
    if split == "train":
        meta_path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        meta_path = Config.VAL_METADATA_PATH
    else:
        meta_path = Config.TEST_METADATA_PATH

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df_meta = pd.read_csv(meta_path)

    # Get unique drive/phone combinations
    # For test set, tripId is unique, but we process by drive_id/phone_name
    unique_trips = df_meta[["drive_id", "phone_name"]].drop_duplicates()

    if Config.DEBUG:
        unique_trips = unique_trips.head(Config.DEBUG_DRIVE_COUNT)

    dataset = []

    print(f"Loading {split} data for {len(unique_trips)} trips...")

    for _, row in unique_trips.iterrows():
        drive_data = prepare_drive_data(
            row["drive_id"], row["phone_name"], df_meta, load_cached_data
        )

        if drive_data is not None:
            # Add identifiers
            drive_data["drive_id"] = row["drive_id"]
            drive_data["phone_name"] = row["phone_name"]
            dataset.append(drive_data)

    return dataset
