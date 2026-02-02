import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import ECEF_to_ENU, WGS84_to_ECEF


def load_drive_data(drive_id, phone_name, input_dir=Config.INPUT_DIR):
    """
    Load raw GNSS, IMU, and Ground Truth data for a specific drive and phone.

    Args:
        drive_id (str): The drive identifier.
        phone_name (str): The phone model name.
        input_dir (str): Root directory of the input data.

    Returns:
        tuple: (df_gnss, df_imu, df_gt) or (None, None, None) if files missing.
    """
    # Construct paths based on directory structure
    # Try train first, then test
    base_path = os.path.join(input_dir, "train", drive_id, phone_name)
    if not os.path.exists(base_path):
        base_path = os.path.join(input_dir, "test", drive_id, phone_name)

    if not os.path.exists(base_path):
        return None, None, None

    gnss_path = os.path.join(base_path, "device_gnss.csv")
    imu_path = os.path.join(base_path, "device_imu.csv")
    gt_path = os.path.join(base_path, "ground_truth.csv")

    # Load GNSS
    if os.path.exists(gnss_path):
        # Read specific columns to save memory
        # We need raw measurements for residuals and WLS position for baseline
        df_gnss = pd.read_csv(gnss_path)
    else:
        df_gnss = pd.DataFrame()

    # Load IMU
    if os.path.exists(imu_path):
        df_imu = pd.read_csv(imu_path)
    else:
        df_imu = pd.DataFrame()

    # Load GT (only exists for train)
    if os.path.exists(gt_path):
        df_gt = pd.read_csv(gt_path)
    else:
        df_gt = pd.DataFrame()

    return df_gnss, df_imu, df_gt


def compute_geometry_features(df_gnss):
    """
    Compute physics-based features: Residual Forces and Geometry Matrix diagonals.

    Args:
        df_gnss (pd.DataFrame): Raw GNSS data with WLS positions and Satellite positions.

    Returns:
        pd.DataFrame: Aggregated epoch-level features.
    """
    if df_gnss.empty:
        return pd.DataFrame()

    # Filter for reasonable signals
    df = df_gnss[df_gnss["Cn0DbHz"] >= Config.CN0_THRESHOLD_DBHZ].copy()

    # Constants
    c = Config.LIGHT_SPEED

    # 1. Calculate Geometric Distance (Range) from WLS to Sat
    # WLS position is the linearization point
    dx = df["SvPositionXEcefMeters"] - df["WlsPositionXEcefMeters"]
    dy = df["SvPositionYEcefMeters"] - df["WlsPositionYEcefMeters"]
    dz = df["SvPositionZEcefMeters"] - df["WlsPositionZEcefMeters"]
    dist = np.sqrt(dx**2 + dy**2 + dz**2)

    # 2. Line of Sight Vectors (Unit vectors)
    # u = (Sat - User) / |Sat - User|
    ux = dx / dist
    uy = dy / dist
    uz = dz / dist

    # 3. Corrected Pseudorange
    # RawPr = GeometricRange + c * (dt_rx - dt_sat) + Ion + Trop + Noise
    # We approximate the residual: r = RawPr - GeometricRange - c * dt_rx
    # Note: BiasNanos is the receiver clock bias.
    # We use the provided BiasNanos. FullBiasNanos is usually large and handled by the chipset time.
    # BiasNanos is the sub-second part.

    # Handle NaNs in Bias
    bias_meters = df["BiasNanos"].fillna(0) * 1e-9 * c

    # Simple residual approximation
    # We ignore satellite clock bias here as it is usually corrected in SvPosition or small relative to multipath in urban canyons
    # Ideally: residual = RawPseudorange - dist + c * sat_clk - c * rx_clk - iono - tropo
    # We simplify to: residual = RawPseudorange - dist - c * rx_clk
    # This leaves: Geometry Error + Multipath + Noise + Atmos
    residuals = df["RawPseudorangeMeters"] - dist - bias_meters

    # 4. Project Residuals to ENU frame
    # We need WLS Lat/Lon for rotation matrix
    # Since WLS is ECEF, we convert to Lat/Lon for the rotation
    # This is expensive per row. We can use the Ground Truth Lat/Lon if available,
    # but for test set we don't have it. We must use WLS ECEF -> WGS84.
    # For efficiency, we can assume a local tangent plane is stable over small areas,
    # but let's do it properly using vector operations.

    # Rotate ECEF unit vectors (ux, uy, uz) to ENU (ue, un, uu)
    # We need a reference Lat/Lon for the rotation.
    # We can approximate using the first valid WLS position of the epoch or the drive.
    # However, vectorizing ECEF_to_ENU requires Lat/Lon.
    # Let's assume the WLS position is close enough to define the local tangent plane.
    # We'll use a simplified spherical approximation for rotation to save time or use the util if fast enough.

    # Vectorized rotation is tricky without pre-computed Lat/Lon.
    # Let's skip precise ENU rotation per satellite and compute ECEF Forces first: F_x, F_y, F_z
    # Then rotate the aggregate force to ENU at the epoch level.

    # Force Vector (ECEF) = residual * unit_vector
    # Weighting: We can weight by Cn0. Higher Cn0 -> Trust more.
    # weight = 10 ** (Cn0 / 10)
    w = 10 ** (df["Cn0DbHz"] / 10.0)

    Fx = residuals * ux * w
    Fy = residuals * uy * w
    Fz = residuals * uz * w

    # Geometry Matrix Elements (Outer product of LOS vectors weighted)
    # G = sum(w * u * u.T)
    Gxx = w * ux * ux
    Gyy = w * uy * uy
    Gzz = w * uz * uz

    # Assign back to dataframe
    df["Fx"] = Fx
    df["Fy"] = Fy
    df["Fz"] = Fz
    df["Gxx"] = Gxx
    df["Gyy"] = Gyy
    df["Gzz"] = Gzz
    df["weight"] = w

    # 5. Aggregate to Epoch Level
    # Group by utcTimeMillis
    agg_funcs = {
        "Cn0DbHz": ["mean", "std", "min", "max"],
        "Svid": "count",
        "SvElevationDegrees": ["mean", "std"],
        "BiasUncertaintyNanos": "mean",
        "DriftUncertaintyNanosPerSecond": "mean",
        "Fx": "sum",
        "Fy": "sum",
        "Fz": "sum",
        "Gxx": "sum",
        "Gyy": "sum",
        "Gzz": "sum",
        "weight": "sum",
        # We also need the WLS position of the epoch to rotate ECEF forces to ENU
        "WlsPositionXEcefMeters": "first",
        "WlsPositionYEcefMeters": "first",
        "WlsPositionZEcefMeters": "first",
    }

    df_epoch = df.groupby("utcTimeMillis").agg(agg_funcs)

    # Flatten columns
    df_epoch.columns = [
        "_".join(col).strip() if isinstance(col, tuple) else col
        for col in df_epoch.columns.values
    ]
    df_epoch.rename(columns={"Svid_count": "Svid_count"}, inplace=True)

    # Normalize Forces by total weight
    w_sum = df_epoch["weight_sum"] + 1e-9
    df_epoch["Fx_sum"] /= w_sum
    df_epoch["Fy_sum"] /= w_sum
    df_epoch["Fz_sum"] /= w_sum

    # Rotate Average Force from ECEF to ENU using WLS position
    # We need to convert WLS ECEF to Lat/Lon for rotation
    # We can use a simplified conversion or the utility.
    # Since we have many epochs, we iterate or vectorize. Vectorization is preferred.
    # However, `library.utils.ECEF_to_WGS84` supports numpy arrays.

    wls_x = df_epoch["WlsPositionXEcefMeters_first"].values
    wls_y = df_epoch["WlsPositionYEcefMeters_first"].values
    wls_z = df_epoch["WlsPositionZEcefMeters_first"].values

    # Convert WLS to Geodetic for rotation reference
    # This might be slow if millions of points, but for one drive it's fine.
    # Note: This utility handles arrays.
    from library.utils import ECEF_to_WGS84  # Import inside to ensure visibility

    lat_wls, lon_wls, _ = ECEF_to_WGS84(wls_x, wls_y, wls_z)

    # Now rotate Force vector (Fx, Fy, Fz) to ENU
    # E = -sin(lon)*Fx + cos(lon)*Fy
    # N = -sin(lat)cos(lon)*Fx - sin(lat)sin(lon)*Fy + cos(lat)*Fz
    # U = cos(lat)cos(lon)*Fx + cos(lat)sin(lon)*Fy + sin(lat)*Fz

    sin_lat = np.sin(np.radians(lat_wls))
    cos_lat = np.cos(np.radians(lat_wls))
    sin_lon = np.sin(np.radians(lon_wls))
    cos_lon = np.cos(np.radians(lon_wls))

    fx = df_epoch["Fx_sum"].values
    fy = df_epoch["Fy_sum"].values
    fz = df_epoch["Fz_sum"].values

    fe = -sin_lon * fx + cos_lon * fy
    fn = -sin_lat * cos_lon * fx - sin_lat * sin_lon * fy + cos_lat * fz
    fu = cos_lat * cos_lon * fx + cos_lat * sin_lon * fy + sin_lat * fz

    df_epoch["Force_E"] = fe
    df_epoch["Force_N"] = fn
    df_epoch["Force_U"] = fu

    # Rename Geometry diagonals
    df_epoch.rename(
        columns={"Gxx_sum": "G_xx", "Gyy_sum": "G_yy", "Gzz_sum": "G_zz"}, inplace=True
    )

    # Clean up
    drop_cols = [
        c
        for c in df_epoch.columns
        if "Wls" in c or "weight" in c or "_sum" in c and "Force" not in c
    ]
    # Keep WLS for later merging/target calculation if needed, but we usually merge on timestamp
    # Actually we need WLS position to compute targets (GT - WLS).
    # So keep WLS position columns but rename them to standard
    df_epoch.rename(
        columns={
            "WlsPositionXEcefMeters_first": "Wls_X",
            "WlsPositionYEcefMeters_first": "Wls_Y",
            "WlsPositionZEcefMeters_first": "Wls_Z",
        },
        inplace=True,
    )

    return df_epoch


def compute_imu_features(df_imu):
    """
    Compute aggregated IMU features.
    """
    if df_imu.empty:
        return pd.DataFrame()

    # Calculate magnitude
    df_imu["magnitude"] = np.sqrt(
        df_imu["MeasurementX"] ** 2
        + df_imu["MeasurementY"] ** 2
        + df_imu["MeasurementZ"] ** 2
    )

    # Pivot or filter by type
    # We focus on UncalAccel and UncalGyro
    acc = df_imu[df_imu["MessageType"] == "UncalAccel"].copy()
    gyro = df_imu[df_imu["MessageType"] == "UncalGyro"].copy()

    # Aggregate by utcTimeMillis
    # Note: IMU timestamps might need rounding to match GNSS
    # GNSS utcTimeMillis is integer ms. IMU is also ms.
    # We round to nearest second or keep ms?
    # The task requires 1Hz output. GNSS is usually 1Hz. IMU is ~100Hz.
    # We group IMU by the nearest GNSS epoch.
    # Assuming GNSS epochs are roughly every 1000ms.

    # Simple approach: Round to nearest second (1000ms)
    acc["epoch_ts"] = np.round(acc["utcTimeMillis"] / 1000.0) * 1000.0
    gyro["epoch_ts"] = np.round(gyro["utcTimeMillis"] / 1000.0) * 1000.0

    acc_agg = (
        acc.groupby("epoch_ts")["magnitude"]
        .agg(["mean", "std"])
        .add_prefix("Accel_Mag_")
    )
    gyro_agg = (
        gyro.groupby("epoch_ts")["magnitude"]
        .agg(["mean", "std"])
        .add_prefix("Gyro_Mag_")
    )

    imu_feats = pd.concat([acc_agg, gyro_agg], axis=1)
    imu_feats.index.name = "utcTimeMillis"
    return imu_feats


def process_drive(drive_id, phone_name, df_gnss, df_imu, df_gt=None):
    """
    Process a single drive: feature engineering + target alignment.
    """
    # 1. GNSS Features
    gnss_feats = compute_geometry_features(df_gnss)
    if gnss_feats.empty:
        return None

    # 2. IMU Features
    imu_feats = compute_imu_features(df_imu)

    # 3. Merge Features
    # Join on index (utcTimeMillis)
    # Note: IMU index is float (rounded), GNSS is int.
    gnss_feats.index = gnss_feats.index.astype(float)

    # Use merge_asof or simple join?
    # Since we rounded IMU to nearest 1000ms, and GNSS is usually on the second, join should work.
    features = gnss_feats.join(imu_feats, how="left")

    # Fill missing IMU with 0 or mean
    features.fillna(0, inplace=True)

    # 4. Calculate Targets (if GT exists)
    if df_gt is not None and not df_gt.empty:
        # Align GT to Features
        # GT has UnixTimeMillis.
        # Features index is utcTimeMillis.

        # Reset index to make merging easier
        features = features.reset_index()

        # Merge
        # We use inner join because we only train on labeled epochs
        merged = pd.merge(
            features,
            df_gt,
            left_on="utcTimeMillis",
            right_on="UnixTimeMillis",
            how="inner",
        )

        # Compute ENU Residuals (Target)
        # Target = GT - WLS
        # We need to convert GT (Lat/Lon) to ECEF, then subtract Wls_X/Y/Z, then rotate to ENU.

        gt_lat = merged["LatitudeDegrees"].values
        gt_lon = merged["LongitudeDegrees"].values
        # GT Altitude is often noisy or missing (NaN).
        # For ENU rotation, we need an anchor. We can use WLS position as anchor.
        # For Target calculation, we need GT Altitude. If missing, assume WLS Altitude?
        # Or just compute Horizontal Error?
        # The metric is horizontal. But to get accurate East/North, we need 3D diff.
        # Let's use WLS altitude for GT if GT alt is NaN, or just 0.
        gt_alt = merged["AltitudeMeters"].fillna(0).values

        gt_x, gt_y, gt_z = WGS84_to_ECEF(gt_lat, gt_lon, gt_alt)

        wls_x = merged["Wls_X"].values
        wls_y = merged["Wls_Y"].values
        wls_z = merged["Wls_Z"].values

        # Delta ECEF
        dx = gt_x - wls_x
        dy = gt_y - wls_y
        dz = gt_z - wls_z

        # Rotate to ENU using WLS as anchor (since that's what we want to correct)
        # We need Lat/Lon of WLS.
        from library.utils import ECEF_to_WGS84, ECEF_to_ENU

        wls_lat, wls_lon, _ = ECEF_to_WGS84(wls_x, wls_y, wls_z)

        # Vectorized ENU conversion manually to avoid loop overhead
        sin_lat = np.sin(np.radians(wls_lat))
        cos_lat = np.cos(np.radians(wls_lat))
        sin_lon = np.sin(np.radians(wls_lon))
        cos_lon = np.cos(np.radians(wls_lon))

        res_e = -sin_lon * dx + cos_lon * dy
        res_n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
        # res_u = ... (not needed for horizontal metric)

        merged["res_E"] = res_e
        merged["res_N"] = res_n

        # Add Speed for feature
        if "SpeedMps" in df_gt.columns:
            merged["WLS_SpeedMps"] = df_gt[
                "SpeedMps"
            ]  # Using GT speed as proxy for now, ideally derive from Doppler
        else:
            merged["WLS_SpeedMps"] = 0.0

        return merged

    else:
        # Test mode: No GT
        features = features.reset_index()
        # Add placeholder targets
        features["res_E"] = 0.0
        features["res_N"] = 0.0
        features["WLS_SpeedMps"] = 0.0  # Placeholder

        # We need to preserve tripId for submission
        features["tripId"] = f"{drive_id}-{phone_name}"
        features["UnixTimeMillis"] = features["utcTimeMillis"]

        return features


def load_dataset(split="train", load_cached_data=True):
    """
    Main entry point to load the dataset for a specific split.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, try to load from parquet cache.

    Returns:
        pd.DataFrame: The processed dataset ready for training/inference.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"{split}_dataset.parquet")

    # 1. Try Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split} data from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    # 2. Generate Data
    print(f"Generating {split} data from raw files...")

    # Load Metadata
    if split == "train":
        meta_path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        meta_path = Config.VAL_METADATA_PATH
    else:
        meta_path = Config.TEST_METADATA_PATH

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    meta_df = pd.read_csv(meta_path)

    # Get unique drive-phone pairs
    # In test, we iterate unique tripIds, but efficient loading requires grouping by drive
    # meta_df has columns: tripId, drive_id, phone_name

    groups = meta_df[["drive_id", "phone_name"]].drop_duplicates()

    all_data = []

    for _, row in groups.iterrows():
        drive_id = row["drive_id"]
        phone_name = row["phone_name"]

        # Load Raw
        df_gnss, df_imu, df_gt = load_drive_data(drive_id, phone_name)

        if df_gnss is None or df_gnss.empty:
            continue

        # Process
        processed_df = process_drive(drive_id, phone_name, df_gnss, df_imu, df_gt)

        if processed_df is not None:
            # Filter to only rows present in metadata (requested timestamps)
            # This is crucial for Test set, and good for Train to match GT
            target_timestamps = meta_df[
                (meta_df["drive_id"] == drive_id)
                & (meta_df["phone_name"] == phone_name)
            ]["UnixTimeMillis"].values

            # Filter
            processed_df = processed_df[
                processed_df["UnixTimeMillis"].isin(target_timestamps)
            ].copy()

            # Add identifiers
            processed_df["drive_id"] = drive_id
            processed_df["phone_name"] = phone_name

            all_data.append(processed_df)

    if not all_data:
        raise ValueError(f"No data generated for split {split}")

    final_df = pd.concat(all_data, ignore_index=True)

    # 3. Save Cache
    print(f"Saving {split} data to cache: {cache_path}")
    final_df.to_parquet(cache_path, index=False)

    return final_df
