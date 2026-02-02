import os
import pandas as pd
import numpy as np
import library.config as C
import library.utils as U


def load_drive_data(drive_id, phone_name, gnss_path, gt_path=None):
    """
    Loads GNSS and Ground Truth data (if available) for a specific drive and phone.

    Args:
        drive_id (str): Drive identifier.
        phone_name (str): Phone model name.
        gnss_path (str): Relative path to GNSS file.
        gt_path (str): Relative path to Ground Truth file (optional).

    Returns:
        tuple: (gnss_df, gt_df)
    """
    # Load GNSS Data
    full_gnss_path = os.path.join(C.INPUT_DIR, gnss_path)
    if not os.path.exists(full_gnss_path):
        # Fallback or error handling
        print(f"Warning: GNSS file not found at {full_gnss_path}")
        return None, None

    gnss_df = pd.read_csv(full_gnss_path)

    # Load Ground Truth Data if path is provided
    gt_df = None
    if gt_path:
        full_gt_path = os.path.join(C.INPUT_DIR, gt_path)
        if os.path.exists(full_gt_path):
            gt_df = pd.read_csv(full_gt_path)

    return gnss_df, gt_df


def aggregate_features(gnss_df):
    """
    Aligns raw measurements to 1Hz timestamps and computes statistics.
    Also extracts/converts WLS baseline positions.

    Args:
        gnss_df (pd.DataFrame): Raw GNSS data.

    Returns:
        pd.DataFrame: Aggregated features with 1Hz resolution.
    """
    if "utcTimeMillis" not in gnss_df.columns:
        return pd.DataFrame()

    # 1. Aggregation of Signal Features
    grouped = gnss_df.groupby("utcTimeMillis")

    # Define aggregation dictionary based on config
    agg_dict = {}
    for col in C.RAW_GNSS_COLS:
        if col in gnss_df.columns:
            agg_dict[col] = C.AGGREGATION_MAP.get(col, ["mean"])

    # Perform aggregation
    if not agg_dict:
        return pd.DataFrame()

    agg_df = grouped.agg(agg_dict)

    # Flatten MultiIndex columns
    agg_df.columns = [f"{col}_{stat}" for col, stat in agg_df.columns]

    # 2. Derived Features
    # Satellite Count
    agg_df["sat_count"] = grouped.size()

    # 3. Extract Baseline WLS Position
    # We take the mean of WLS positions for the epoch (they are usually identical per epoch)
    wls_cols = [
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]
    if all(c in gnss_df.columns for c in wls_cols):
        wls_pos = grouped[wls_cols].mean()

        # Convert ECEF to Geodetic (Lat, Lon, Alt)
        lats, lons, alts = U.ecef_to_geodetic(
            wls_pos["WlsPositionXEcefMeters"].values,
            wls_pos["WlsPositionYEcefMeters"].values,
            wls_pos["WlsPositionZEcefMeters"].values,
        )

        agg_df["wls_lat"] = lats
        agg_df["wls_lon"] = lons
        agg_df["wls_alt"] = alts
    else:
        # Should not happen given dataset description, but handle gracefully
        agg_df["wls_lat"] = np.nan
        agg_df["wls_lon"] = np.nan
        agg_df["wls_alt"] = np.nan

    # Reset index to make UnixTimeMillis a column
    agg_df = agg_df.reset_index().rename(columns={"utcTimeMillis": "UnixTimeMillis"})

    return agg_df


def prepare_sequences(metadata_path, load_cached_data=True, split_name="train"):
    """
    Main processing function. Reads metadata, processes each trip,
    and returns a list of DataFrames ready for the model.

    Args:
        metadata_path (str): Path to the metadata CSV.
        load_cached_data (bool): Whether to load from cache if available.
        split_name (str): 'train', 'val', or 'test' for cache naming.

    Returns:
        list[pd.DataFrame]: List of processed dataframes (one per trip).
    """
    if not os.path.exists(metadata_path):
        print(f"Metadata file not found: {metadata_path}")
        return []

    df_meta = pd.read_csv(metadata_path)

    # Identify unique trips
    # For test, tripId is unique. For train, we group by drive_id and phone_name.
    if "tripId" in df_meta.columns:
        trips = df_meta[["drive_id", "phone_name", "tripId"]].drop_duplicates()
    else:
        # Create a dummy tripId for train/val
        trips = df_meta[["drive_id", "phone_name"]].drop_duplicates()
        trips["tripId"] = trips["drive_id"] + "-" + trips["phone_name"]

    processed_sequences = []

    print(f"Processing {len(trips)} trips for {split_name}...")

    for _, row in trips.iterrows():
        drive_id = row["drive_id"]
        phone_name = row["phone_name"]
        trip_id = row["tripId"]

        # Cache File Path
        # Use safe filename
        safe_trip_id = trip_id.replace("/", "_")
        cache_file = os.path.join(C.WORKING_DIR, f"{safe_trip_id}_{split_name}.parquet")

        # 1. Try Load Cache
        if load_cached_data and os.path.exists(cache_file):
            try:
                df_seq = pd.read_parquet(cache_file)
                processed_sequences.append(df_seq)
                continue
            except Exception:
                pass  # Fall through to recompute

        # 2. Compute from Scratch
        # Get metadata rows for this trip
        trip_meta = df_meta[
            (df_meta["drive_id"] == drive_id) & (df_meta["phone_name"] == phone_name)
        ].copy()

        if trip_meta.empty:
            continue

        # Extract paths
        gnss_path = trip_meta.iloc[0]["gnss_path"]

        # Load Raw GNSS
        gnss_df, _ = load_drive_data(drive_id, phone_name, gnss_path, None)

        if gnss_df is None or gnss_df.empty:
            continue

        # Aggregate Features
        agg_df = aggregate_features(gnss_df)

        if agg_df.empty:
            continue

        # Merge logic depends on split
        if split_name in ["train", "val"]:
            # Inner join to keep only labeled timestamps
            merged_df = pd.merge(agg_df, trip_meta, on="UnixTimeMillis", how="inner")

            # Compute Targets (ENU Residuals)
            # We assume GT altitude is approx same as WLS altitude for horizontal error calc
            # if AltitudeMeters is not in metadata.
            if (
                "wls_lat" in merged_df.columns
                and "LatitudeDegrees" in merged_df.columns
            ):
                # Use WLS altitude for both to isolate horizontal error
                ref_alt = merged_df["wls_alt"].values

                e, n, u = U.geodetic_to_enu(
                    merged_df["LatitudeDegrees"].values,
                    merged_df["LongitudeDegrees"].values,
                    ref_alt,
                    merged_df["wls_lat"].values,
                    merged_df["wls_lon"].values,
                    merged_df["wls_alt"].values,
                )

                merged_df["dLat_meters"] = n  # North
                merged_df["dLon_meters"] = e  # East

            final_df = merged_df

        else:  # Test
            # Left join to keep all submission timestamps
            merged_df = pd.merge(trip_meta, agg_df, on="UnixTimeMillis", how="left")

            # Interpolate missing WLS positions
            wls_cols = ["wls_lat", "wls_lon", "wls_alt"]
            if all(c in merged_df.columns for c in wls_cols):
                # Interpolate, then forward/back fill to handle edges
                merged_df[wls_cols] = merged_df[wls_cols].interpolate(
                    method="linear", limit_direction="both"
                )
                merged_df[wls_cols] = merged_df[wls_cols].ffill().bfill()

            # Fill missing features with 0
            feature_cols = [
                c for c in agg_df.columns if c not in ["UnixTimeMillis"] + wls_cols
            ]
            merged_df[feature_cols] = merged_df[feature_cols].fillna(0)

            final_df = merged_df

        # Add Phone ID
        final_df["phone_idx"] = C.PHONE_NAME_TO_IDX.get(phone_name, 0)

        # Save to Cache
        try:
            final_df.to_parquet(cache_file, index=False)
        except Exception as e:
            print(f"Warning: Could not save cache for {trip_id}: {e}")

        processed_sequences.append(final_df)

    return processed_sequences
