import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import warnings

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
INPUT_DIR = "./input"
OUTPUT_DIR = "./working/idea_13"
SUBMISSION_DIR = "./submission"
METADATA_DIR = "./metadata"

# Data Settings
SECTORS = 4  # Number of azimuthal sectors (quadrants)
SECTOR_WIDTH = 360 / SECTORS

# Model Hyperparameters
LGB_PARAMS = {
    "objective": "mae",
    "boosting_type": "gbdt",
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "num_leaves": 64,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

# Training Settings
N_FOLDS = 5
EARLY_STOPPING_ROUNDS = 50
VERBOSE_EVAL = 50

# Constants for Physics
LIGHT_SPEED = 299792458.0  # m/s
OMEGA_EARTH = 7.2921151467e-5  # rad/s

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates distance in meters between two lat/lon points."""
    R = 6371000  # Radius of Earth in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def ecef_to_enu(x, y, z, lat0, lon0, h0):
    """Converts ECEF coordinates to ENU coordinates relative to a reference point."""
    # WGS84 ellipsoid constants
    a = 6378137.0
    b = 6356752.3142
    f = (a - b) / a
    e_sq = f * (2 - f)

    lamb = np.radians(lat0)
    phi = np.radians(lon0)
    s = np.sin(lamb)
    N = a / np.sqrt(1 - e_sq * s * s)

    sin_lambda = np.sin(lamb)
    cos_lambda = np.cos(lamb)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    x0 = (h0 + N) * cos_lambda * cos_phi
    y0 = (h0 + N) * cos_lambda * sin_phi
    z0 = (h0 + (1 - e_sq) * N) * sin_lambda

    xd = x - x0
    yd = y - y0
    zd = z - z0

    xEast = -sin_phi * xd + cos_phi * yd
    yNorth = -sin_lambda * cos_phi * xd - sin_lambda * sin_phi * yd + cos_lambda * zd
    zUp = cos_lambda * cos_phi * xd + cos_lambda * sin_phi * yd + sin_lambda * zd

    return xEast, yNorth, zUp


def get_sector(azimuth):
    """Maps azimuth (0-360) to a sector index."""
    return (azimuth // SECTOR_WIDTH).astype(int) % SECTORS


# =============================================================================
# DATA PROCESSING
# =============================================================================


def process_gnss_data(gnss_df):
    """
    Extracts physics-based residuals from GNSS data.
    """
    # Filter valid signals
    # Keep signals with valid WLS position and Satellite position
    mask = (
        gnss_df["WlsPositionXEcefMeters"].notna()
        & gnss_df["SvPositionXEcefMeters"].notna()
        & gnss_df["RawPseudorangeMeters"].notna()
    )
    df = gnss_df[mask].copy()

    # Calculate Geometric Distance (Sat to WLS)
    # Note: Simple Euclidean distance in ECEF. Sagnac effect is small for residuals but could be added.
    dx = df["SvPositionXEcefMeters"] - df["WlsPositionXEcefMeters"]
    dy = df["SvPositionYEcefMeters"] - df["WlsPositionYEcefMeters"]
    dz = df["SvPositionZEcefMeters"] - df["WlsPositionZEcefMeters"]
    dist = np.sqrt(dx**2 + dy**2 + dz**2)

    # Correct Pseudorange
    # CorrectedPR = RawPR + SatClockBias - Ionosphere - Troposphere - ISRB
    # Note: BiasNanos in raw logs is receiver clock bias, but we treat it as common mode error later.
    # SvClockBiasMeters is typically negative (correction to apply).
    # We assume standard sign convention: PR_corrected = PR_raw + SatClk + ...

    # Fill NaNs in corrections with 0
    df["SvClockBiasMeters"] = df["SvClockBiasMeters"].fillna(0)
    df["IsrbMeters"] = df["IsrbMeters"].fillna(0)
    df["IonosphericDelayMeters"] = df["IonosphericDelayMeters"].fillna(0)
    df["TroposphericDelayMeters"] = df["TroposphericDelayMeters"].fillna(0)

    corrected_pr = (
        df["RawPseudorangeMeters"]
        + df["SvClockBiasMeters"]
        - df["IsrbMeters"]
        - df["IonosphericDelayMeters"]
        - df["TroposphericDelayMeters"]
    )

    # Calculate Raw Residual
    # Residual = CorrectedPR - GeometricDistance
    # This includes Receiver Clock Bias (meters)
    raw_residual = corrected_pr - dist

    # Remove Receiver Clock Bias (Common Mode Error)
    # We estimate it as the median residual per epoch (per phone, per timestamp)
    # utcTimeMillis identifies the epoch
    epoch_bias = raw_residual.groupby(df["utcTimeMillis"]).transform("median")
    df["pr_residual"] = raw_residual - epoch_bias

    # Doppler Residuals
    # Range Rate approx = projection of relative velocity
    # We assume user velocity is negligible compared to sat velocity for residual check,
    # or rely on the fact that we remove common mode drift.
    # A better approx uses SvVelocity projected onto Line-of-Sight.

    # Line of Sight Unit Vector
    ux = dx / dist
    uy = dy / dist
    uz = dz / dist

    # Projected Satellite Velocity (approximated range rate)
    # Note: Relative velocity = V_sat - V_user. Assuming V_user=0 for feature extraction baseline.
    range_rate_est = (
        df["SvVelocityXEcefMetersPerSecond"] * ux
        + df["SvVelocityYEcefMetersPerSecond"] * uy
        + df["SvVelocityZEcefMetersPerSecond"] * uz
    )

    # Corrected Pseudorange Rate
    # Rate = Measured - SatClockDrift
    df["SvClockDriftMetersPerSecond"] = df["SvClockDriftMetersPerSecond"].fillna(0)
    corrected_rate = (
        df["PseudorangeRateMetersPerSecond"] + df["SvClockDriftMetersPerSecond"]
    )

    raw_doppler_res = corrected_rate - range_rate_est

    # Remove Receiver Clock Drift (Common Mode)
    drift_bias = raw_doppler_res.groupby(df["utcTimeMillis"]).transform("median")
    df["doppler_residual"] = raw_doppler_res - drift_bias

    return df


def aggregate_features(gnss_df, imu_df=None):
    """
    Aggregates GNSS features by sector and merges with IMU stats.
    """
    # Sector calculation
    gnss_df["sector"] = get_sector(gnss_df["SvAzimuthDegrees"].fillna(0))

    # Features to aggregate
    aggs = {
        "pr_residual": ["mean", "std", "max", "min"],
        "doppler_residual": ["mean", "std"],
        "Cn0DbHz": ["mean", "max"],
        "SvElevationDegrees": ["mean"],
    }

    # Pivot by sector
    # We group by utcTimeMillis and sector
    pivot_df = gnss_df.groupby(["utcTimeMillis", "sector"]).agg(aggs)

    # Flatten columns: e.g., pr_residual_mean_sector0
    pivot_df.columns = [f"{c[0]}_{c[1]}" for c in pivot_df.columns]
    pivot_df = pivot_df.unstack(level="sector")

    # Flatten again: feature_stat_sectorX
    pivot_df.columns = [f"{c[0]}_s{c[1]}" for c in pivot_df.columns]

    # Global aggregates (across all sectors)
    global_aggs = gnss_df.groupby("utcTimeMillis").agg(
        {
            "pr_residual": ["mean", "std"],
            "Cn0DbHz": ["mean", "count"],  # count is num satellites
            "SvElevationDegrees": ["mean"],
        }
    )
    global_aggs.columns = [f"global_{c[0]}_{c[1]}" for c in global_aggs.columns]

    result = pd.concat([pivot_df, global_aggs], axis=1)

    # Fill missing sectors with NaN or 0?
    # LightGBM handles NaN, but 0 might be safer for "no signal in this sector".
    # Let's leave as NaN for LightGBM to decide.

    # IMU Aggregation (if provided)
    if imu_df is not None:
        # Calculate magnitude for Accel
        acc_mask = imu_df["MessageType"] == "UncalAccel"
        acc_df = imu_df[acc_mask].copy()
        acc_df["mag"] = np.sqrt(
            acc_df["MeasurementX"] ** 2
            + acc_df["MeasurementY"] ** 2
            + acc_df["MeasurementZ"] ** 2
        )

        # Aggregate to nearest second
        # IMU is high freq, GNSS is 1Hz. We group IMU by rounding timestamp to nearest epoch?
        # Or just taking mean over the window.
        # Simple approach: round utcTimeMillis to nearest 1000
        acc_df["epoch_ts"] = np.round(acc_df["utcTimeMillis"] / 1000) * 1000

        imu_feats = (
            acc_df.groupby("epoch_ts")["mag"]
            .agg(["mean", "std"])
            .add_prefix("imu_acc_")
        )

        # Merge
        result.index = result.index.astype(float)
        # GNSS timestamps are exact, IMU are approximate.
        # We need to match GNSS utcTimeMillis to IMU epoch_ts
        # GNSS utcTimeMillis usually ends in various digits, but roughly 1Hz.
        # Let's round GNSS index for merging too? No, keep GNSS exact, merge on rounded.
        result["merge_key"] = np.round(result.index / 1000) * 1000
        result = (
            result.reset_index()
            .merge(imu_feats, left_on="merge_key", right_index=True, how="left")
            .drop(columns=["merge_key"])
            .set_index("utcTimeMillis")
        )

    return result


def get_dataset(metadata_path, load_cached_data=True, split="train"):
    """
    Loads raw data based on metadata, processes it, and returns a feature dataframe.
    Implements caching.
    """
    cache_file = os.path.join(OUTPUT_DIR, f"{split}_features.parquet")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached {split} data from {cache_file}...")
        return pd.read_parquet(cache_file)

    print(f"Processing {split} data from scratch...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    meta_df = pd.read_csv(metadata_path)

    # To avoid OOM, process by drive and append
    features_list = []
    targets_list = []

    unique_drives = meta_df["drive_id"].unique()

    for drive_id in unique_drives:
        drive_meta = meta_df[meta_df["drive_id"] == drive_id]

        # There might be multiple phones per drive
        for phone_name in drive_meta["phone_name"].unique():
            subset = drive_meta[drive_meta["phone_name"] == phone_name]
            if subset.empty:
                continue

            # Load Sensor Data
            # Paths in metadata are relative to input dir
            gnss_rel_path = subset.iloc[0]["gnss_path"]
            imu_rel_path = subset.iloc[0]["imu_path"]

            gnss_path = os.path.join(INPUT_DIR, gnss_rel_path)
            imu_path = os.path.join(INPUT_DIR, imu_rel_path)

            if not os.path.exists(gnss_path):
                print(f"Warning: GNSS file missing: {gnss_path}")
                continue

            gnss_df = pd.read_csv(gnss_path)
            imu_df = pd.read_csv(imu_path) if os.path.exists(imu_path) else None

            # Process Features
            proc_gnss = process_gnss_data(gnss_df)
            drive_feats = aggregate_features(proc_gnss, imu_df)

            # Align with Metadata (Ground Truth / Submission Targets)
            # Metadata has the exact timestamps we need
            # We merge features onto metadata
            subset_indexed = subset.set_index("UnixTimeMillis")

            # Inner join to keep only rows where we have both GT/Target and Features
            # Note: GNSS timestamps might not match exactly with GT timestamps.
            # Usually they align well, but sometimes a small tolerance is needed.
            # For this task, exact match on UnixTimeMillis usually works for 'device_gnss.csv' rows
            # vs 'ground_truth.csv' rows if they come from the same log.
            # However, 'device_gnss.csv' has 'utcTimeMillis'.

            merged = subset_indexed.join(drive_feats, how="inner")

            if merged.empty:
                continue

            # Calculate Targets (ENU Residuals) if training
            if split in ["train", "val"]:
                # We need WLS positions to calculate residuals
                # The WLS positions are in the GNSS file. We need to aggregate them to 1Hz
                # (e.g. mean of signal WLS positions for that epoch) or take the first one.
                # Since WLS is per-epoch, all signals in one epoch share the same WLS position.
                wls_pos = proc_gnss.groupby("utcTimeMillis")[
                    [
                        "WlsPositionXEcefMeters",
                        "WlsPositionYEcefMeters",
                        "WlsPositionZEcefMeters",
                    ]
                ].first()

                # Join WLS to merged
                merged = merged.join(wls_pos, how="inner")

                # Convert GT (Lat/Lon) to ECEF
                # We assume Altitude is 0 or use WLS altitude if GT altitude missing?
                # Metadata doesn't have Altitude. We can use WLS altitude as reference or 0.
                # Better: Convert WLS ECEF to Lat/Lon, then compute difference in meters directly.
                # Or: Use flat earth approximation.

                # Let's use Flat Earth approx for residuals (Lat/Lon diff to Meters)
                # dLat_m = (Lat_GT - Lat_WLS) * 111320
                # dLon_m = (Lon_GT - Lon_WLS) * 111320 * cos(Lat)

                # First convert WLS ECEF to Lat/Lon/Alt
                # Simplified conversion or use library? We can't install libraries.
                # We can implement ecef2lla.

                # Vectorized ECEF to LLA
                x = merged["WlsPositionXEcefMeters"].values
                y = merged["WlsPositionYEcefMeters"].values
                z = merged["WlsPositionZEcefMeters"].values

                # Simple approximation or iterative
                # WGS84
                a = 6378137.0
                e = 8.1819190842622e-2

                b = np.sqrt(a**2 * (1 - e**2))
                ep = np.sqrt((a**2 - b**2) / b**2)
                p = np.sqrt(x**2 + y**2)
                th = np.arctan2(a * z, b * p)
                lon_wls = np.arctan2(y, x)
                lat_wls = np.arctan2(
                    z + ep**2 * b * np.sin(th) ** 3, p - e**2 * a * np.cos(th) ** 3
                )

                lat_wls_deg = np.degrees(lat_wls)
                lon_wls_deg = np.degrees(lon_wls)

                # Calculate Targets
                dLat = merged["LatitudeDegrees"] - lat_wls_deg
                dLon = merged["LongitudeDegrees"] - lon_wls_deg

                # Handle Lon wrapping? Usually not an issue for small residuals in same drive.

                target_N = dLat * 111320.0
                target_E = (
                    dLon * 111320.0 * np.cos(np.radians(merged["LatitudeDegrees"]))
                )

                merged["target_E"] = target_E
                merged["target_N"] = target_N

                # Add WLS Lat/Lon to df for reconstruction later (validation)
                merged["wls_lat"] = lat_wls_deg
                merged["wls_lon"] = lon_wls_deg

            else:
                # Test set: We need WLS Lat/Lon for reconstruction
                wls_pos = proc_gnss.groupby("utcTimeMillis")[
                    [
                        "WlsPositionXEcefMeters",
                        "WlsPositionYEcefMeters",
                        "WlsPositionZEcefMeters",
                    ]
                ].first()
                merged = merged.join(wls_pos, how="inner")

                x = merged["WlsPositionXEcefMeters"].values
                y = merged["WlsPositionYEcefMeters"].values
                z = merged["WlsPositionZEcefMeters"].values

                a = 6378137.0
                e = 8.1819190842622e-2
                b = np.sqrt(a**2 * (1 - e**2))
                ep = np.sqrt((a**2 - b**2) / b**2)
                p = np.sqrt(x**2 + y**2)
                th = np.arctan2(a * z, b * p)
                lon_wls = np.arctan2(y, x)
                lat_wls = np.arctan2(
                    z + ep**2 * b * np.sin(th) ** 3, p - e**2 * a * np.cos(th) ** 3
                )

                merged["wls_lat"] = np.degrees(lat_wls)
                merged["wls_lon"] = np.degrees(lon_wls)

            features_list.append(merged)

    if not features_list:
        raise ValueError(f"No data processed for {split} split!")

    full_df = pd.concat(features_list)

    # Save to cache
    full_df.to_parquet(cache_file)
    print(f"Saved {split} data to {cache_file}. Shape: {full_df.shape}")

    return full_df


# =============================================================================
# MODEL TRAINING & INFERENCE
# =============================================================================


def train_model(train_df, val_df):
    """
    Trains LightGBM models for East and North residuals using GroupKFold.
    """
    feature_cols = [
        c
        for c in train_df.columns
        if c.startswith(("pr_", "doppler_", "Cn0", "SvEl", "global_", "imu_"))
    ]
    print(f"Training with {len(feature_cols)} features.")

    # Prepare Data
    X_train = train_df[feature_cols]
    y_train_E = train_df["target_E"]
    y_train_N = train_df["target_N"]
    groups = train_df["drive_id"]

    # We use the provided validation set as a hold-out for final scoring,
    # but for training the ensemble, we cross-validate on the training set.
    # Actually, the prompt implies using the val set for evaluation.
    # Let's train on Train, Eval on Val for simplicity and robustness,
    # OR use CV on Train and ensemble.
    # Strategy: Train 5 folds on TrainDF (grouped by drive). Ensemble predicts on ValDF to score.

    models_E = []
    models_N = []

    gkf = GroupKFold(n_splits=N_FOLDS)

    # --- Train East Models ---
    print("\nTraining East Component...")
    oof_preds_E = np.zeros(len(train_df))

    for fold, (trn_idx, dev_idx) in enumerate(gkf.split(X_train, y_train_E, groups)):
        X_t, y_t = X_train.iloc[trn_idx], y_train_E.iloc[trn_idx]
        X_d, y_d = X_train.iloc[dev_idx], y_train_E.iloc[dev_idx]

        lgb_train = lgb.Dataset(X_t, y_t)
        lgb_eval = lgb.Dataset(X_d, y_d, reference=lgb_train)

        model = lgb.train(
            LGB_PARAMS,
            lgb_train,
            valid_sets=[lgb_eval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
                lgb.log_evaluation(VERBOSE_EVAL),
            ],
        )
        models_E.append(model)
        oof_preds_E[dev_idx] = model.predict(X_d)

    print(f"Train OOF MAE East: {mean_absolute_error(y_train_E, oof_preds_E)}")

    # --- Train North Models ---
    print("\nTraining North Component...")
    oof_preds_N = np.zeros(len(train_df))

    for fold, (trn_idx, dev_idx) in enumerate(gkf.split(X_train, y_train_N, groups)):
        X_t, y_t = X_train.iloc[trn_idx], y_train_N.iloc[trn_idx]
        X_d, y_d = X_train.iloc[dev_idx], y_train_N.iloc[dev_idx]

        lgb_train = lgb.Dataset(X_t, y_t)
        lgb_eval = lgb.Dataset(X_d, y_d, reference=lgb_train)

        model = lgb.train(
            LGB_PARAMS,
            lgb_train,
            valid_sets=[lgb_eval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
                lgb.log_evaluation(VERBOSE_EVAL),
            ],
        )
        models_N.append(model)
        oof_preds_N[dev_idx] = model.predict(X_d)

    print(f"Train OOF MAE North: {mean_absolute_error(y_train_N, oof_preds_N)}")

    return models_E, models_N, feature_cols


def inference(models_E, models_N, test_df, feature_cols):
    """
    Generates predictions using the ensemble.
    """
    X_test = test_df[feature_cols]

    # Predict East
    preds_E = []
    for model in models_E:
        preds_E.append(model.predict(X_test))
    # Pixel-wise Median
    pred_E_final = np.median(np.column_stack(preds_E), axis=1)

    # Predict North
    preds_N = []
    for model in models_N:
        preds_N.append(model.predict(X_test))
    pred_N_final = np.median(np.column_stack(preds_N), axis=1)

    return pred_E_final, pred_N_final


def run_pipeline(load_cached_data=True):
    # 1. Load Data
    train_df = get_dataset(
        os.path.join(METADATA_DIR, "train_metadata.csv"), load_cached_data, "train"
    )
    val_df = get_dataset(
        os.path.join(METADATA_DIR, "val_metadata.csv"), load_cached_data, "val"
    )
    test_df = get_dataset(
        os.path.join(METADATA_DIR, "test_metadata.csv"), load_cached_data, "test"
    )

    # 2. Train
    models_E, models_N, feats = train_model(train_df, val_df)

    # 3. Validate on Hold-out Set
    print("\nValidating on Hold-out Set...")
    val_pred_E, val_pred_N = inference(models_E, models_N, val_df, feats)

    mae_E = mean_absolute_error(val_df["target_E"], val_pred_E)
    mae_N = mean_absolute_error(val_df["target_N"], val_pred_N)
    print(f"Validation MAE East: {mae_E}")
    print(f"Validation MAE North: {mae_N}")
    print(
        f"Validation Mean Distance Error (approx): {(mae_E + mae_N)/2 * np.sqrt(2)}"
    )  # Rough estimate

    # 4. Predict on Test
    print("\nGenerating Test Predictions...")
    test_pred_E, test_pred_N = inference(models_E, models_N, test_df, feats)

    # 5. Reconstruct Trajectory
    # New Lat = WLS_Lat + dN / 111320
    # New Lon = WLS_Lon + dE / (111320 * cos(Lat))

    test_df["pred_lat"] = test_df["wls_lat"] + test_pred_N / 111320.0
    test_df["pred_lon"] = test_df["wls_lon"] + test_pred_E / (
        111320.0 * np.cos(np.radians(test_df["wls_lat"]))
    )

    # 6. Save Submission
    submission = test_df[["tripId", "UnixTimeMillis", "pred_lat", "pred_lon"]].copy()
    submission.columns = [
        "tripId",
        "UnixTimeMillis",
        "LatitudeDegrees",
        "LongitudeDegrees",
    ]

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")
