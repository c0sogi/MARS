import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import ecef_to_geodetic, geodetic_to_enu


def _load_and_aggregate_gnss(gnss_path):
    """
    Loads raw GNSS data, aligns timestamps to 1Hz, and performs distribution-aware aggregation.
    """
    if not os.path.exists(gnss_path):
        raise FileNotFoundError(f"GNSS file not found: {gnss_path}")

    # Define columns to read: Features + Svid (for count) + WLS Baseline
    # Using a set to avoid duplicates
    wls_cols = [
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]
    cols_to_read = list(set(Config.GNSS_COLS + ["Svid"] + wls_cols))

    # Load data
    try:
        df = pd.read_csv(gnss_path, usecols=lambda c: c in cols_to_read)
    except ValueError:
        # Fallback if usecols fails (e.g. missing columns), read all then filter
        df = pd.read_csv(gnss_path)
        missing_cols = [c for c in cols_to_read if c not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns in {gnss_path}: {missing_cols}")

    # Check if WLS columns exist; if not, we can't compute targets/baseline
    if not all(c in df.columns for c in wls_cols):
        print(f"Warning: Critical WLS columns missing in {gnss_path}. Skipping drive.")
        return pd.DataFrame()

    # Align timestamps: Round utcTimeMillis to nearest second (UnixTimeMillis)
    # utcTimeMillis is in ms.
    df["UnixTimeMillis"] = (np.round(df["utcTimeMillis"] / 1000) * 1000).astype(
        np.int64
    )

    # Prepare Aggregation Dictionary
    # We construct a dictionary for named aggregation
    agg_dict = {}

    # 1. Satellite Count
    agg_dict["SatCount"] = ("Svid", "count")

    # 2. Distribution Statistics for Features
    for col, stats in Config.AGGREGATION_SPECS.items():
        if col not in df.columns:
            continue

        for stat in stats:
            feat_name = f"{col}_{stat}"

            if stat == "q25":
                func = lambda x: x.quantile(0.25)
            elif stat == "q75":
                func = lambda x: x.quantile(0.75)
            else:
                func = stat

            agg_dict[feat_name] = (col, func)

    # 3. WLS Baseline (Take first value per epoch, as it's an epoch-level estimate)
    for col in wls_cols:
        agg_dict[col] = (col, "first")

    # Perform GroupBy Aggregation
    try:
        df_agg = df.groupby("UnixTimeMillis").agg(**agg_dict)
    except Exception as e:
        print(f"Error during aggregation for {gnss_path}: {e}")
        return pd.DataFrame()

    # Reset index to make UnixTimeMillis a column
    df_agg = df_agg.reset_index()

    return df_agg


def _compute_targets(df_features, gt_path):
    """
    Loads ground truth, merges with aggregated features, and computes ENU residuals.
    Target = GroundTruth_ENU - WLS_ENU (relative to WLS position).
    """
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    df_gt = pd.read_csv(gt_path)

    # Cite debug_lesson_4: Normalize Timestamp Precision Before Merging Time-Series Data
    # Create a rounded timestamp column in GT to match the 1Hz resolution of features
    df_gt["UnixTimeMillis_rounded"] = (
        np.round(df_gt["UnixTimeMillis"] / 1000) * 1000
    ).astype(np.int64)

    # Merge Features and Ground Truth on aligned timestamp
    # Inner join ensures we only train on epochs with valid GT
    df_merged = pd.merge(
        df_features,
        df_gt[
            [
                "UnixTimeMillis_rounded",
                "LatitudeDegrees",
                "LongitudeDegrees",
                "AltitudeMeters",
            ]
        ],
        left_on="UnixTimeMillis",
        right_on="UnixTimeMillis_rounded",
        how="inner",
    )

    if df_merged.empty:
        return pd.DataFrame()

    # 1. Convert WLS ECEF to Geodetic (Lat, Lon, Alt)
    # This serves as our local reference point for ENU conversion
    wls_x = df_merged["WlsPositionXEcefMeters"].values
    wls_y = df_merged["WlsPositionYEcefMeters"].values
    wls_z = df_merged["WlsPositionZEcefMeters"].values

    wls_lat, wls_lon, wls_alt = ecef_to_geodetic(wls_x, wls_y, wls_z)

    # Store WLS Geodetic for reference (useful for reconstruction/debugging)
    df_merged["WlsLat"] = wls_lat
    df_merged["WlsLon"] = wls_lon
    df_merged["WlsAlt"] = wls_alt

    # 2. Compute Target Residuals (ENU)
    # We calculate the ENU coordinates of the Ground Truth *relative* to the WLS position.
    # This vector (East, North) represents the correction the model needs to predict.
    gt_lat = df_merged["LatitudeDegrees"].values
    gt_lon = df_merged["LongitudeDegrees"].values
    gt_alt = df_merged["AltitudeMeters"].values

    d_east, d_north, d_up = geodetic_to_enu(
        gt_lat, gt_lon, gt_alt, wls_lat, wls_lon, wls_alt
    )

    df_merged["DeltaEast"] = d_east
    df_merged["DeltaNorth"] = d_north
    # DeltaUp is computed but typically not used for 2D horizontal error minimization

    # Drop rows where targets are NaN (due to missing WLS baseline)
    before_len = len(df_merged)
    df_merged = df_merged.dropna(subset=["DeltaEast", "DeltaNorth"])
    after_len = len(df_merged)

    # Cite debug_lesson_5: Validate Data Volume After Filtering
    if after_len == 0 and before_len > 0:
        print(
            f"Warning: All rows dropped due to missing WLS/Target data in {gt_path}. "
            "Check if WLS columns in GNSS file contain valid data."
        )
        return pd.DataFrame()

    return df_merged


def preprocess_drive(
    drive_id, phone_name, gnss_path, gt_path=None, load_cached_data=True
):
    """
    Preprocesses a single drive.

    Args:
        drive_id (str): Identifier for the drive.
        phone_name (str): Identifier for the phone.
        gnss_path (str): Relative path to raw GNSS csv.
        gt_path (str, optional): Relative path to ground truth csv. If None, runs in inference mode.
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        pd.DataFrame: Processed dataframe with features and (optional) targets.
    """
    # 1. Construct Cache Path
    # Sanitize filename
    safe_drive = drive_id.replace("/", "_").replace("\\", "_")
    safe_phone = phone_name.replace("/", "_").replace("\\", "_")
    mode = "train" if gt_path else "test"
    cache_filename = f"{safe_drive}_{safe_phone}_{mode}.parquet"
    cache_file_path = os.path.join(Config.CACHE_DIR, cache_filename)

    # 2. Try Loading Cache
    if load_cached_data and os.path.exists(cache_file_path):
        try:
            df = pd.read_parquet(cache_file_path)
            return df
        except Exception as e:
            print(
                f"Failed to load cache for {drive_id} {phone_name}: {e}. Recomputing..."
            )

    # 3. Compute from Scratch
    full_gnss_path = os.path.join(Config.INPUT_DIR, gnss_path)

    # Load and Aggregate Features
    try:
        df_agg = _load_and_aggregate_gnss(full_gnss_path)
    except FileNotFoundError:
        print(f"Warning: GNSS file missing for {drive_id} {phone_name}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error processing GNSS for {drive_id} {phone_name}: {e}")
        return pd.DataFrame()

    if df_agg.empty:
        return pd.DataFrame()

    # Compute Targets (Train Mode) or Prepare Inference Data (Test Mode)
    if gt_path:
        full_gt_path = os.path.join(Config.INPUT_DIR, gt_path)
        try:
            df_result = _compute_targets(df_agg, full_gt_path)
        except Exception as e:
            print(f"Error computing targets for {drive_id} {phone_name}: {e}")
            return pd.DataFrame()
    else:
        # Inference Mode: Calculate WLS Geodetic for later reconstruction
        wls_x = df_agg["WlsPositionXEcefMeters"].values
        wls_y = df_agg["WlsPositionYEcefMeters"].values
        wls_z = df_agg["WlsPositionZEcefMeters"].values
        wls_lat, wls_lon, wls_alt = ecef_to_geodetic(wls_x, wls_y, wls_z)

        df_agg["WlsLat"] = wls_lat
        df_agg["WlsLon"] = wls_lon
        df_agg["WlsAlt"] = wls_alt
        df_result = df_agg

    # Add Metadata columns
    df_result["drive_id"] = drive_id
    df_result["phone_name"] = phone_name

    # 4. Save to Cache
    if not df_result.empty:
        try:
            df_result.to_parquet(cache_file_path, index=False)
        except Exception as e:
            print(f"Failed to save cache for {drive_id} {phone_name}: {e}")

    return df_result


def load_dataset(metadata_path, load_cached_data=True):
    """
    Loads and processes all drives listed in the metadata file.

    Args:
        metadata_path (str): Path to the metadata CSV (train, val, or test).
        load_cached_data (bool): Whether to use cached parquet files.

    Returns:
        pd.DataFrame: Concatenated dataframe of all processed drives.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)

    # Identify unique drive-phone pairs to process
    # Metadata contains one row per timestamp, but we process per file.
    unique_drives = df_meta[["drive_id", "phone_name", "gnss_path"]].drop_duplicates()

    # Check if targets exist in metadata (heuristic to determine if this is train/val or test)
    is_train = "LatitudeDegrees" in df_meta.columns

    processed_dfs = []

    print(f"Processing {len(unique_drives)} unique drives from {metadata_path}...")

    for _, row in unique_drives.iterrows():
        drive_id = row["drive_id"]
        phone_name = row["phone_name"]
        gnss_path = row["gnss_path"]

        gt_path = None
        if is_train:
            # Reconstruct GT path based on directory structure
            # train/[drive_id]/[phone_name]/ground_truth.csv
            gt_path = os.path.join("train", drive_id, phone_name, "ground_truth.csv")

        df_drive = preprocess_drive(
            drive_id, phone_name, gnss_path, gt_path, load_cached_data
        )

        if not df_drive.empty:
            processed_dfs.append(df_drive)

    if not processed_dfs:
        print("Warning: No data processed.")
        return pd.DataFrame()

    full_df = pd.concat(processed_dfs, ignore_index=True)
    return full_df
