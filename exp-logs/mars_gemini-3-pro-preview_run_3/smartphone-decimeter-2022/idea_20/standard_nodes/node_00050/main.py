import os
import sys
import shutil
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
from scipy.stats import pearsonr

# Import library classes
from library.data_loader import GnssDataset
from library.kinematics import KinematicsEngine
from library.model import ResidualRegressor
from library.optimizer import TrajectoryOptimizer
from library.utils import CoordinateTransformer, MetricCalculator, IOHelper

# Configuration
SEED = 42
VALIDATION_THRESHOLD = 4.160290813847215
CACHE_DIR = "./working/idea_20/"
os.makedirs(CACHE_DIR, exist_ok=True)

# Set seeds
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def get_kinematics(mode):
    """
    Extracts kinematics (Doppler velocity & TDCP displacement) for a given mode.
    """
    print(f"Extracting Kinematics for {mode}...")

    # Load raw data (cached if available)
    loader = GnssDataset(mode=mode)
    df_raw = loader.load(load_cached_data=True)

    engine = KinematicsEngine()
    kinematics_list = []

    # Process each trip
    trips = df_raw["tripId"].unique()
    for trip in trips:
        trip_df = df_raw[df_raw["tripId"] == trip].copy()
        # KinematicsEngine handles caching internally per trip
        k_df = engine.process_trip(trip_df, load_cached_data=True)
        if not k_df.empty:
            kinematics_list.append(k_df)

    if not kinematics_list:
        return pd.DataFrame()

    return pd.concat(kinematics_list, ignore_index=True)


def train_and_validate():
    """
    Trains the LightGBM models on the training set and evaluates on the validation set.
    """
    print("\n=== Training & Validation Phase ===")

    # 1. Prepare Data (Features & Targets)
    # ResidualRegressor.prepare_data handles feature engineering and caching
    regressor = ResidualRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=31)

    df_train = regressor.prepare_data(mode="train", load_cached_data=True)
    df_val = regressor.prepare_data(mode="val", load_cached_data=True)

    # 2. Define Features
    exclude_cols = [
        "tripId",
        "utcTimeMillis",
        "drive_id",
        "phone_name",
        "LatitudeDegrees",
        "LongitudeDegrees",
        "AltitudeMeters",
        "SpeedMps",
        "AccuracyMeters",
        "BearingDegrees",
        "target_east",
        "target_north",
        "target_up",
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
        "gnss_path",
        "imu_path",
        "gt_path",
        "MessageType",
        "MeasurementX",
        "MeasurementY",
        "MeasurementZ",
    ]

    feature_cols = [
        c
        for c in df_train.columns
        if c not in exclude_cols
        and df_train[c].dtype in [np.float64, np.float32, np.int64, np.int32]
    ]

    # Handle NaNs/Infs
    for df in [df_train, df_val]:
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        df[feature_cols] = df[feature_cols].fillna(0)

    print(f"Training with {len(feature_cols)} features.")

    # 3. Train Models (East & North)
    # Using simple fit on full train set for maximum data, validated on hold-out val set
    X_train = df_train[feature_cols]
    y_east_train = df_train["target_east"]
    y_north_train = df_train["target_north"]

    # Check for GPU
    device_type = "cpu"
    if torch.cuda.is_available():
        # LightGBM GPU support requires specific build, often not default in pip.
        # We try to pass 'gpu' but fallback to 'cpu' is handled by LGBM usually or we catch error.
        # For safety in this environment, we stick to cpu or 'cuda' if supported.
        # We'll use default (cpu) to ensure stability as 'gpu' param might crash if not compiled.
        pass

    print("Training East Model...")
    model_e = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        objective="mae",
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model_e.fit(X_train, y_east_train)

    print("Training North Model...")
    model_n = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        objective="mae",
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model_n.fit(X_train, y_north_train)

    # 4. Predict on Validation (Anchors)
    print("Predicting on Validation Set...")
    X_val = df_val[feature_cols]
    pred_e = model_e.predict(X_val)
    pred_n = model_n.predict(X_val)

    # Convert ENU predictions to Lat/Lon (Anchors)
    # We need reference WLS positions
    wls_x = df_val["WlsPositionXEcefMeters"].values
    wls_y = df_val["WlsPositionYEcefMeters"].values
    wls_z = df_val["WlsPositionZEcefMeters"].values

    ref_lat, ref_lon, ref_alt = CoordinateTransformer.ecef_to_wgs84(wls_x, wls_y, wls_z)

    # ENU -> ECEF -> WGS84
    pred_x, pred_y, pred_z = CoordinateTransformer.enu_to_ecef(
        pred_e, pred_n, np.zeros_like(pred_e), ref_lat, ref_lon, ref_alt
    )
    pred_lat, pred_lon, _ = CoordinateTransformer.ecef_to_wgs84(pred_x, pred_y, pred_z)

    df_val_pred = df_val[["tripId", "utcTimeMillis"]].copy()
    df_val_pred.rename(columns={"utcTimeMillis": "UnixTimeMillis"}, inplace=True)
    df_val_pred["LatitudeDegrees"] = pred_lat
    df_val_pred["LongitudeDegrees"] = pred_lon

    # 5. Optimize Validation Trajectories
    # Load kinematics for Val
    df_val_kin = get_kinematics("val")

    optimizer = TrajectoryOptimizer()
    df_val_opt = optimizer.optimize_all(df_val_pred, df_val_kin, load_cached_data=True)

    # 6. Evaluation
    # Prepare GT for scoring
    df_val_gt = df_val[
        ["tripId", "utcTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    ].copy()
    df_val_gt.rename(columns={"utcTimeMillis": "UnixTimeMillis"}, inplace=True)

    score = MetricCalculator.calc_score(df_val_opt, df_val_gt)
    print(f"Final Validation Metric: {score}")

    # 7. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate distance error per point
    merged = pd.merge(
        df_val_opt,
        df_val_gt,
        on=["tripId", "UnixTimeMillis"],
        suffixes=("_pred", "_gt"),
    )
    dists = MetricCalculator.haversine_distance(
        merged["LatitudeDegrees_pred"],
        merged["LongitudeDegrees_pred"],
        merged["LatitudeDegrees_gt"],
        merged["LongitudeDegrees_gt"],
    )

    # Merge features back to calculate correlation
    # Need to match indices or merge on keys. X_val aligns with df_val.
    # df_val_opt might have different order if optimize_all changed it, but optimize_all processes by trip.
    # Let's merge error back to df_val
    merged_analysis = pd.merge(
        df_val,
        merged[["tripId", "UnixTimeMillis", "dist"]].rename(
            columns={"UnixTimeMillis": "utcTimeMillis"}
        ),
        on=["tripId", "utcTimeMillis"],
    )

    print("Correlation between Error Distance and Features:")
    correlations = []
    for col in feature_cols:
        if merged_analysis[col].std() > 0:
            corr, _ = pearsonr(merged_analysis["dist"], merged_analysis[col])
            correlations.append((col, corr))

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for name, corr in correlations[:10]:
        print(f"  {name}: {corr:.4f}")

    return score, model_e, model_n, feature_cols


def inference(model_e, model_n, feature_cols):
    """
    Generates submission for the test set.
    """
    print("\n=== Inference Phase ===")

    # 1. Prepare Test Data
    regressor = ResidualRegressor()
    df_test = regressor.prepare_data(mode="test", load_cached_data=True)

    # Handle NaNs
    df_test[feature_cols] = df_test[feature_cols].replace([np.inf, -np.inf], np.nan)
    df_test[feature_cols] = df_test[feature_cols].fillna(0)

    # 2. Predict Anchors
    X_test = df_test[feature_cols]
    pred_e = model_e.predict(X_test)
    pred_n = model_n.predict(X_test)

    # Convert to Lat/Lon
    wls_x = df_test["WlsPositionXEcefMeters"].values
    wls_y = df_test["WlsPositionYEcefMeters"].values
    wls_z = df_test["WlsPositionZEcefMeters"].values

    ref_lat, ref_lon, ref_alt = CoordinateTransformer.ecef_to_wgs84(wls_x, wls_y, wls_z)

    pred_x, pred_y, pred_z = CoordinateTransformer.enu_to_ecef(
        pred_e, pred_n, np.zeros_like(pred_e), ref_lat, ref_lon, ref_alt
    )
    pred_lat, pred_lon, _ = CoordinateTransformer.ecef_to_wgs84(pred_x, pred_y, pred_z)

    df_test_pred = pd.DataFrame(
        {
            "tripId": df_test["tripId"],
            "UnixTimeMillis": df_test["utcTimeMillis"],
            "LatitudeDegrees": pred_lat,
            "LongitudeDegrees": pred_lon,
        }
    )

    # 3. Optimize Test Trajectories
    df_test_kin = get_kinematics("test")

    optimizer = TrajectoryOptimizer()
    df_test_opt = optimizer.optimize_all(
        df_test_pred, df_test_kin, load_cached_data=True
    )

    # 4. Save Submission
    os.makedirs("./submission", exist_ok=True)
    sub_path = "./submission/submission.csv"
    df_test_opt.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")


if __name__ == "__main__":
    score, model_e, model_n, feats = train_and_validate()

    if score < VALIDATION_THRESHOLD:
        print(
            f"Validation score {score:.4f} passed threshold {VALIDATION_THRESHOLD}. Proceeding to inference."
        )
        inference(model_e, model_n, feats)
    else:
        print(
            f"Validation score {score:.4f} did not pass threshold {VALIDATION_THRESHOLD}. Skipping inference."
        )
