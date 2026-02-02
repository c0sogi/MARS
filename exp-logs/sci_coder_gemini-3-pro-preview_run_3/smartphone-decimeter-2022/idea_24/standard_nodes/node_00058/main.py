import os
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from library.config import Config
from library.data_io import load_metadata
from library.model import prepare_dataset, LightGBMRegressor
from library.utils import haversine_distance, ecef_to_llh, enu_to_ecef, llh_to_ecef
from library.optimization import run_global_optimization

# Set random seed
np.random.seed(Config.SEED)


def clean_stale_artifacts():
    """
    Removes stale cache files and submission outputs to ensure a fresh run.
    Cite debug_lesson_3: Invalidate Data Caches When Modifying Data Processing Logic
    """
    print("Cleaning stale artifacts...")
    # Remove dataset caches
    dataset_pattern = os.path.join(Config.WORKING_DIR, "dataset_*.parquet")
    for f in glob.glob(dataset_pattern):
        try:
            os.remove(f)
            print(f"Removed stale cache: {f}")
        except OSError as e:
            print(f"Error removing {f}: {e}")

    # Remove submission file
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    if os.path.exists(submission_path):
        try:
            os.remove(submission_path)
            print(f"Removed stale submission: {submission_path}")
        except OSError as e:
            print(f"Error removing submission: {e}")


def get_feature_cols(df):
    """Identifies feature columns for training."""
    exclude_cols = [
        "tripId",
        "UnixTimeMillis",
        "LatitudeDegrees",
        "LongitudeDegrees",
        "drive_id",
        "phone_name",
        "gnss_path",
        "imu_path",
        "gt_path",
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
        "Target_E",
        "Target_N",
    ]
    return [c for c in df.columns if c not in exclude_cols]


def train_and_validate():
    print("Loading dataset...")
    # prepare_dataset("train") loads both train and val metadata and merges them
    # Cite debug_lesson_4: Preserve Row Cardinality During Inference
    df_all = prepare_dataset("train", load_cached_data=False)

    if df_all.empty:
        raise ValueError("No training data loaded.")

    # Load split definitions to separate them
    train_meta = load_metadata("train")
    val_meta = load_metadata("val")

    train_drives = set(train_meta["drive_id"].unique())
    val_drives = set(val_meta["drive_id"].unique())

    print(f"Total drives: {df_all['drive_id'].nunique()}")
    print(f"Train drives: {len(train_drives)}")
    print(f"Val drives: {len(val_drives)}")

    # Split data
    df_train = df_all[df_all["drive_id"].isin(train_drives)].copy()
    df_val = df_all[df_all["drive_id"].isin(val_drives)].copy()

    print(f"Train samples: {len(df_train)}")
    print(f"Val samples: {len(df_val)}")

    feature_cols = get_feature_cols(df_train)
    print(f"Training with {len(feature_cols)} features.")

    # Train Models
    # We use the LightGBMRegressor wrapper but manage the training call manually
    # to ensure we save models to the correct location for the optimizer to pick up later.

    # Train East
    print("\n--- Training East Model ---")
    model_e = LightGBMRegressor("Target_E")
    # We use the drive_id as groups for CV within the training set,
    # though for this baseline we are just training on the full train split
    # and validating on the hold-out val split.
    # To utilize the class method, we pass the train set.
    model_e.train_group_kfold(
        df_train[feature_cols], df_train["Target_E"], df_train["drive_id"], n_splits=5
    )

    # Train North
    print("\n--- Training North Model ---")
    model_n = LightGBMRegressor("Target_N")
    model_n.train_group_kfold(
        df_train[feature_cols], df_train["Target_N"], df_train["drive_id"], n_splits=5
    )

    # Inference on Validation Set
    print("\n--- Validating ---")
    X_val = df_val[feature_cols]
    pred_e = model_e.predict(X_val)
    pred_n = model_n.predict(X_val)

    # Reconstruct Absolute Positions
    val_lats = []
    val_lons = []

    wls_x = df_val["WlsPositionXEcefMeters"].values
    wls_y = df_val["WlsPositionYEcefMeters"].values
    wls_z = df_val["WlsPositionZEcefMeters"].values

    for i in range(len(df_val)):
        wx, wy, wz = wls_x[i], wls_y[i], wls_z[i]
        de, dn = pred_e[i], pred_n[i]

        # Reference WLS LLH
        ref_lat, ref_lon, ref_alt = ecef_to_llh(wx, wy, wz)

        # Convert predicted ENU offset to ECEF (assuming dU=0)
        dx, dy, dz = enu_to_ecef(de, dn, 0, ref_lat, ref_lon, ref_alt)

        # Convert back to LLH
        plat, plon, _ = ecef_to_llh(dx, dy, dz)

        val_lats.append(plat)
        val_lons.append(plon)

    df_val["Pred_Lat"] = val_lats
    df_val["Pred_Lon"] = val_lons

    # Compute Metric
    df_val["error_dist"] = haversine_distance(
        df_val["LatitudeDegrees"],
        df_val["LongitudeDegrees"],
        df_val["Pred_Lat"],
        df_val["Pred_Lon"],
    )

    phone_scores = []
    for phone, group in df_val.groupby("phone_name"):
        p50 = np.percentile(group["error_dist"], 50)
        p95 = np.percentile(group["error_dist"], 95)
        score = (p50 + p95) / 2
        phone_scores.append(score)

    final_metric = np.mean(phone_scores)
    print(f"Final Validation Metric: {final_metric}")

    return df_val, final_metric, feature_cols


def failure_analysis(df_val, feature_cols):
    print("\n--- Failure Analysis ---")
    # Calculate correlation between error distance and features
    correlations = {}
    for col in feature_cols:
        if df_val[col].std() > 0:
            corr, _ = pearsonr(df_val["error_dist"], df_val[col])
            correlations[col] = corr

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 Features correlated with Error Magnitude:")
    for name, corr in sorted_corr[:10]:
        print(f"{name:<30}: {corr:.4f}")


def main():
    # 0. Clean Stale Artifacts
    clean_stale_artifacts()

    # 1. Train and Validate
    df_val, metric, feature_cols = train_and_validate()

    # 2. Failure Analysis
    failure_analysis(df_val, feature_cols)

    # 3. Submission Logic
    THRESHOLD = 4.160290813847215
    if metric < THRESHOLD:
        print(
            f"\nMetric {metric} < {THRESHOLD}. Proceeding to Test Inference and Optimization..."
        )

        # The run_global_optimization function handles:
        # 1. Generating ML predictions for test set (using models saved in ./working/idea_24/models)
        # 2. Computing Kinematics (TDCP)
        # 3. Optimizing the trajectory
        # 4. Saving submission.csv
        run_global_optimization(load_cached_data=False)

    else:
        print(f"\nMetric {metric} >= {THRESHOLD}. Optimization skipped.")


if __name__ == "__main__":
    main()
