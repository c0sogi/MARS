import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import library.config as config
from library.config import (
    INPUT_DIR,
    CACHE_DIR,
    WINDOW_SIZE,
    SEED,
    TRAIN_CACHE_FILES,
    VAL_CACHE_FILES,
    TEST_CACHE_FILES,
)
from library.utils import ecef_to_lla, get_local_scale_factors


def aggregate_gnss_epochs(gnss_df):
    """
    Aggregates raw GNSS measurements into epoch-level statistics and WLS positions.
    """
    # Define aggregation dictionary
    agg_funcs = {
        "WlsPositionXEcefMeters": "first",
        "WlsPositionYEcefMeters": "first",
        "WlsPositionZEcefMeters": "first",
        "SvElevationDegrees": ["mean", "std"],
        "SvAzimuthDegrees": ["mean", "std"],
        "Cn0DbHz": ["mean", "std"],
        "utcTimeMillis": "first",  # Keep time
    }

    # Filter for existing columns
    available_cols = set(gnss_df.columns)
    agg_funcs = {k: v for k, v in agg_funcs.items() if k in available_cols}

    # Group by time
    # Note: device_gnss.csv usually has one unique WLS position per utcTimeMillis
    df_epoch = gnss_df.groupby("utcTimeMillis").agg(agg_funcs)

    # Flatten columns
    df_epoch.columns = [
        "_".join(col).strip() if isinstance(col, tuple) else col
        for col in df_epoch.columns.values
    ]

    # Rename for clarity
    rename_map = {
        "WlsPositionXEcefMeters_first": "x_wls",
        "WlsPositionYEcefMeters_first": "y_wls",
        "WlsPositionZEcefMeters_first": "z_wls",
        "SvElevationDegrees_mean": "el_mean",
        "SvElevationDegrees_std": "el_std",
        "SvAzimuthDegrees_mean": "az_mean",
        "SvAzimuthDegrees_std": "az_std",
        "Cn0DbHz_mean": "cn0_mean",
        "Cn0DbHz_std": "cn0_std",
        "utcTimeMillis_first": "millis",
    }
    # Handle cases where std might be NaN (single satellite) -> fill 0
    df_epoch = df_epoch.rename(columns=rename_map).fillna(0)

    # Sort by time
    df_epoch = df_epoch.sort_values("millis").reset_index(drop=True)

    # Convert WLS ECEF to LLA
    lat, lon, alt = ecef_to_lla(
        df_epoch["x_wls"].values, df_epoch["y_wls"].values, df_epoch["z_wls"].values
    )
    df_epoch["lat_wls"] = lat
    df_epoch["lon_wls"] = lon
    df_epoch["alt_wls"] = alt

    return df_epoch


def create_sliding_windows(df_epoch, target_timestamps, window_size):
    """
    Extracts windows centered on target timestamps.
    Returns Kinematic sequences and Sky Context vectors.
    """
    half_window = window_size // 2

    # Find indices of target timestamps in the epoch dataframe
    # We use searchsorted for efficiency, assuming sorted
    # Note: We need exact matches or very close matches.
    # The provided data usually aligns.

    # Create a lookup series
    # epoch_times = df_epoch['millis'].values
    # target_times = target_timestamps.values

    # Merge to find indices
    df_epoch["original_index"] = df_epoch.index

    # We use a merge to handle potential missing epochs safely
    # target_df just contains the timestamps we need
    target_df = pd.DataFrame({"millis": target_timestamps})
    merged = pd.merge(
        target_df, df_epoch[["millis", "original_index"]], on="millis", how="inner"
    )

    valid_indices = merged["original_index"].values

    kinematic_list = []
    sky_list = []
    valid_mask = []  # To filter targets that don't have enough context

    # Pre-compute metric conversion factors for efficiency?
    # No, we do it per window relative to center.

    # Extract arrays for speed
    lats = df_epoch["lat_wls"].values
    lons = df_epoch["lon_wls"].values
    alts = df_epoch["alt_wls"].values
    cn0s = df_epoch["cn0_mean"].values

    # Sky arrays
    sky_data = df_epoch[
        ["el_mean", "el_std", "az_mean", "az_std", "cn0_mean", "cn0_std"]
    ].values

    N = len(df_epoch)

    for center_idx in valid_indices:
        start_idx = center_idx - half_window
        end_idx = center_idx + half_window + 1  # Slice is exclusive at end

        if start_idx >= 0 and end_idx <= N:
            # 1. Kinematic Stream Construction
            # Extract window
            win_lats = lats[start_idx:end_idx]
            win_lons = lons[start_idx:end_idx]
            win_alts = alts[start_idx:end_idx]
            win_cn0s = cn0s[start_idx:end_idx]

            # Center point
            center_lat = lats[center_idx]
            center_lon = lons[center_idx]
            center_alt = alts[center_idx]

            # Relative coordinates in meters
            # Approx conversion factors
            lat_scale, lon_scale = get_local_scale_factors(center_lat)

            rel_lat_m = (win_lats - center_lat) * lat_scale
            rel_lon_m = (win_lons - center_lon) * lon_scale
            rel_alt_m = win_alts - center_alt

            # Dynamics (Velocity)
            # Use gradient or simple diff. Gradient handles boundaries better.
            # We want diff between steps.
            # Pad the first diff with 0 or repeat? np.gradient is centered.
            # Let's use simple diff and pad first.
            diff_lat = np.diff(rel_lat_m, prepend=rel_lat_m[0])
            diff_lon = np.diff(rel_lon_m, prepend=rel_lon_m[0])
            diff_alt = np.diff(rel_alt_m, prepend=rel_alt_m[0])

            # Stack Kinematic Features: (Window, 7)
            # Features: Lat_rel, Lon_rel, Alt_rel, dLat, dLon, dAlt, Cn0
            kin_seq = np.stack(
                [
                    rel_lat_m,
                    rel_lon_m,
                    rel_alt_m,
                    diff_lat,
                    diff_lon,
                    diff_alt,
                    win_cn0s,
                ],
                axis=1,
            )

            # 2. Sky Context Construction
            # Aggregate stats over the window
            # We simply take the mean of the epoch-level stats over the window
            # to represent the "environmental state"
            win_sky = sky_data[start_idx:end_idx]
            sky_vec = np.mean(win_sky, axis=0)  # (6,)

            kinematic_list.append(kin_seq)
            sky_list.append(sky_vec)
            valid_mask.append(True)
        else:
            valid_mask.append(False)

    if not kinematic_list:
        return np.array([]), np.array([]), np.array(valid_mask, dtype=bool)

    return (
        np.array(kinematic_list),
        np.array(sky_list),
        np.array(valid_mask, dtype=bool),
    )


def process_data(metadata_path, load_cached_data=True, cache_files=None, is_test=False):
    """
    Main processing function. Loads metadata, processes trips, computes features/targets,
    and handles caching.
    """
    # 1. Check Cache
    if load_cached_data and cache_files:
        print(f"Checking cache files: {cache_files}")
        all_exist = all(os.path.exists(f) for f in cache_files.values())
        if all_exist:
            print("Loading data from cache...")
            try:
                X_kin = np.load(cache_files["X_kin"])
                X_sky = np.load(cache_files["X_sky"])
                meta = pd.read_parquet(cache_files["meta"])

                if is_test:
                    return X_kin, X_sky, None, meta
                else:
                    y = np.load(cache_files["y"])
                    return X_kin, X_sky, y, meta
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")
        else:
            print("Cache missing. Recomputing...")

    # 2. Load Metadata
    df_meta = pd.read_csv(metadata_path)

    # 3. Process per Trip
    unique_trips = df_meta["tripId"].unique()

    X_kin_all = []
    X_sky_all = []
    y_all = []
    meta_all = []

    print(f"Processing {len(unique_trips)} trips...")

    for trip_id in tqdm(unique_trips):
        trip_meta = df_meta[df_meta["tripId"] == trip_id].copy()

        # Determine GNSS path (take from first row of trip)
        gnss_rel_path = trip_meta.iloc[0]["gnss_path"]
        gnss_path = os.path.join(INPUT_DIR, gnss_rel_path)

        if not os.path.exists(gnss_path):
            print(f"Warning: GNSS file not found for {trip_id}. Skipping.")
            continue

        # Load and Aggregate GNSS
        gnss_df = pd.read_csv(gnss_path)
        df_epoch = aggregate_gnss_epochs(gnss_df)

        # Create Windows
        target_times = trip_meta["UnixTimeMillis"].values
        X_kin, X_sky, valid_mask = create_sliding_windows(
            df_epoch, target_times, WINDOW_SIZE
        )

        if len(X_kin) == 0:
            continue

        # Filter metadata to valid windows
        trip_meta_valid = trip_meta.iloc[valid_mask].copy()

        # Compute Targets (for Train/Val)
        if not is_test:
            # Get WLS positions for the center timestamps
            # We need to map back from the valid indices logic.
            # create_sliding_windows returns data corresponding to 'valid_mask' true entries in target_times
            # We need the WLS lat/lon for those specific times to compute residuals

            # Re-merge to get WLS for targets
            target_df = pd.DataFrame({"millis": target_times[valid_mask]})
            merged_targets = pd.merge(
                target_df,
                df_epoch[["millis", "lat_wls", "lon_wls"]],
                on="millis",
                how="left",
            )

            wls_lat = merged_targets["lat_wls"].values
            wls_lon = merged_targets["lon_wls"].values
            gt_lat = trip_meta_valid["LatitudeDegrees"].values
            gt_lon = trip_meta_valid["LongitudeDegrees"].values

            # Calculate Scale Factors
            lat_scale, lon_scale = get_local_scale_factors(wls_lat)

            # Calculate Residuals (Target)
            # Target = GT - WLS
            d_lat_deg = gt_lat - wls_lat
            d_lon_deg = gt_lon - wls_lon

            d_lat_m = d_lat_deg * lat_scale
            d_lon_m = d_lon_deg * lon_scale

            y = np.stack([d_lat_m, d_lon_m], axis=1)
            y_all.append(y)

            # Store WLS in metadata for reconstruction later if needed
            trip_meta_valid["wls_lat"] = wls_lat
            trip_meta_valid["wls_lon"] = wls_lon
        else:
            # For test, we still need WLS to reconstruct prediction
            target_df = pd.DataFrame({"millis": target_times[valid_mask]})
            merged_targets = pd.merge(
                target_df,
                df_epoch[["millis", "lat_wls", "lon_wls"]],
                on="millis",
                how="left",
            )
            trip_meta_valid["wls_lat"] = merged_targets["lat_wls"].values
            trip_meta_valid["wls_lon"] = merged_targets["lon_wls"].values

        X_kin_all.append(X_kin)
        X_sky_all.append(X_sky)
        meta_all.append(trip_meta_valid)

    # Concatenate
    if not X_kin_all:
        raise ValueError("No data processed.")

    X_kin_final = np.concatenate(X_kin_all, axis=0)
    X_sky_final = np.concatenate(X_sky_all, axis=0)
    meta_final = pd.concat(meta_all, ignore_index=True)

    if not is_test:
        y_final = np.concatenate(y_all, axis=0)
    else:
        y_final = None

    # 4. Save to Cache
    if cache_files:
        print("Saving data to cache...")
        np.save(cache_files["X_kin"], X_kin_final)
        np.save(cache_files["X_sky"], X_sky_final)
        meta_final.to_parquet(cache_files["meta"], index=False)
        if not is_test:
            np.save(cache_files["y"], y_final)

    return X_kin_final, X_sky_final, y_final, meta_final


def load_data(load_cached_data=True):
    """
    Wrapper to load all splits.
    """
    print("Loading Training Data...")
    train_X_kin, train_X_sky, train_y, train_meta = process_data(
        config.TRAIN_METADATA_PATH,
        load_cached_data=load_cached_data,
        cache_files=config.TRAIN_CACHE_FILES,
        is_test=False,
    )

    print("Loading Validation Data...")
    val_X_kin, val_X_sky, val_y, val_meta = process_data(
        config.VAL_METADATA_PATH,
        load_cached_data=load_cached_data,
        cache_files=config.VAL_CACHE_FILES,
        is_test=False,
    )

    print("Loading Test Data...")
    test_X_kin, test_X_sky, _, test_meta = process_data(
        config.TEST_METADATA_PATH,
        load_cached_data=load_cached_data,
        cache_files=config.TEST_CACHE_FILES,
        is_test=True,
    )

    return (
        (train_X_kin, train_X_sky, train_y, train_meta),
        (val_X_kin, val_X_sky, val_y, val_meta),
        (test_X_kin, test_X_sky, None, test_meta),
    )
