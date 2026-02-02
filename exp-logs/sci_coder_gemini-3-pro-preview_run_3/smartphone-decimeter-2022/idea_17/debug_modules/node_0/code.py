import os
import pandas as pd
import numpy as np
import shutil
import warnings

# Import library components
from library.utils import load_metadata, EcefToEnu, EcefToGeodetic, GeodeticToEcef
from library.features import GeometricFeatureExtractor
from library.odometry import run_odometry_processing
from library.model import ResidualRegressor
from library.trajectory_optimizer import GraphOptimizer, save_submission

# Configuration
SEED = 42
CACHE_DIR = "./working/idea_17"
SUBMISSION_DIR = "./working"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Set seeds
np.random.seed(SEED)
warnings.filterwarnings("ignore")

# --- Monkeypatching GeometricFeatureExtractor ---
# The provided library code for _process_drive has a potential issue where merging on 'utcTimeMillis'
# fails because it's in the index but not a column in the 'features' DataFrame.
# We patch it here to ensure robustness.


def patched_process_drive(
    self, drive_id, phone_name, gnss_path, imu_path, gt_path=None
):
    try:
        df_gnss = pd.read_csv(os.path.join("./input", gnss_path))
        df_imu = pd.read_csv(os.path.join("./input", imu_path))
    except FileNotFoundError:
        print(f"Warning: Missing file for {drive_id}-{phone_name}")
        return None

    # 1. Process IMU
    imu_agg = df_imu.groupby("utcTimeMillis").agg(
        {
            "MeasurementX": ["mean", "std"],
            "MeasurementY": ["mean", "std"],
            "MeasurementZ": ["mean", "std"],
        }
    )
    imu_agg.columns = [f"IMU_{c[0]}_{c[1]}" for c in imu_agg.columns]

    # 2. Process GNSS
    epoch_cols = [
        "utcTimeMillis",
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]
    df_epochs = df_gnss[epoch_cols].drop_duplicates(subset=["utcTimeMillis"]).copy()

    wls_x = df_epochs["WlsPositionXEcefMeters"].values
    wls_y = df_epochs["WlsPositionYEcefMeters"].values
    wls_z = df_epochs["WlsPositionZEcefMeters"].values

    valid_wls = ~np.isnan(wls_x)
    lat_wls = np.zeros_like(wls_x)
    lon_wls = np.zeros_like(wls_x)
    alt_wls = np.zeros_like(wls_x)

    if np.any(valid_wls):
        lat_v, lon_v, alt_v = EcefToGeodetic.transform(
            wls_x[valid_wls], wls_y[valid_wls], wls_z[valid_wls]
        )
        lat_wls[valid_wls] = lat_v
        lon_wls[valid_wls] = lon_v
        alt_wls[valid_wls] = alt_v

    df_epochs["Wls_Lat"] = lat_wls
    df_epochs["Wls_Lon"] = lon_wls
    df_epochs["Wls_Alt"] = alt_wls

    df_gnss = df_gnss.merge(
        df_epochs[["utcTimeMillis", "Wls_Lat", "Wls_Lon", "Wls_Alt"]],
        on="utcTimeMillis",
        how="left",
    )

    dx = df_gnss["SvPositionXEcefMeters"] - df_gnss["WlsPositionXEcefMeters"]
    dy = df_gnss["SvPositionYEcefMeters"] - df_gnss["WlsPositionYEcefMeters"]
    dz = df_gnss["SvPositionZEcefMeters"] - df_gnss["WlsPositionZEcefMeters"]
    dist = np.sqrt(dx**2 + dy**2 + dz**2)

    ux = dx / dist
    uy = dy / dist
    uz = dz / dist

    R = self._get_rotation_matrix(df_gnss["Wls_Lat"].values, df_gnss["Wls_Lon"].values)

    u_ecef = np.stack([ux, uy, uz], axis=1)
    u_enu = np.einsum("ijk,ik->ij", R, u_ecef)

    df_gnss["u_E"] = u_enu[:, 0]
    df_gnss["u_N"] = u_enu[:, 1]
    df_gnss["u_U"] = u_enu[:, 2]

    df_gnss["weight"] = 10 ** (df_gnss["Cn0DbHz"] / 10.0)
    df_gnss["w_uE"] = df_gnss["u_E"] * df_gnss["weight"]
    df_gnss["w_uN"] = df_gnss["u_N"] * df_gnss["weight"]
    df_gnss["w_uU"] = df_gnss["u_U"] * df_gnss["weight"]

    df_gnss["is_L1"] = df_gnss["SignalType"].isin(self.l1_signals)
    df_gnss["is_L5"] = df_gnss["SignalType"].isin(self.l5_signals)

    l1_data = (
        df_gnss[df_gnss["is_L1"]]
        .groupby("utcTimeMillis")[["weight", "w_uE", "w_uN", "w_uU"]]
        .sum()
    )
    l1_counts = df_gnss[df_gnss["is_L1"]].groupby("utcTimeMillis").size()

    l5_data = (
        df_gnss[df_gnss["is_L5"]]
        .groupby("utcTimeMillis")[["weight", "w_uE", "w_uN", "w_uU"]]
        .sum()
    )
    l5_counts = df_gnss[df_gnss["is_L5"]].groupby("utcTimeMillis").size()

    features = pd.DataFrame(index=df_epochs["utcTimeMillis"].unique())
    features.index.name = "utcTimeMillis"

    # --- PATCH START: Ensure index is a column for merging ---
    features = features.reset_index()
    # --- PATCH END ---

    features["L1_SatCount"] = features["utcTimeMillis"].map(l1_counts).fillna(0)
    features["L1_TotalWeight"] = (
        features["utcTimeMillis"].map(l1_data["weight"]).fillna(0)
    )

    # Map other aggregates manually since we reset index
    # (Simplified logic to handle mapping correctly)
    for col in ["w_uE", "w_uN", "w_uU"]:
        features[f"L1_sum_{col}"] = (
            features["utcTimeMillis"].map(l1_data[col]).fillna(0)
        )
        features[f"L5_sum_{col}"] = (
            features["utcTimeMillis"].map(l5_data[col]).fillna(0)
        )

    features["L5_SatCount"] = features["utcTimeMillis"].map(l5_counts).fillna(0)
    features["L5_TotalWeight"] = (
        features["utcTimeMillis"].map(l5_data["weight"]).fillna(0)
    )

    mask_l1 = features["L1_TotalWeight"] > 0
    features.loc[mask_l1, "L1_Proj_E"] = (
        features.loc[mask_l1, "L1_sum_w_uE"] / features.loc[mask_l1, "L1_TotalWeight"]
    )
    features.loc[mask_l1, "L1_Proj_N"] = (
        features.loc[mask_l1, "L1_sum_w_uN"] / features.loc[mask_l1, "L1_TotalWeight"]
    )
    features.loc[mask_l1, "L1_Proj_U"] = (
        features.loc[mask_l1, "L1_sum_w_uU"] / features.loc[mask_l1, "L1_TotalWeight"]
    )

    mask_l5 = features["L5_TotalWeight"] > 0
    features.loc[mask_l5, "L5_Proj_E"] = (
        features.loc[mask_l5, "L5_sum_w_uE"] / features.loc[mask_l5, "L5_TotalWeight"]
    )
    features.loc[mask_l5, "L5_Proj_N"] = (
        features.loc[mask_l5, "L5_sum_w_uN"] / features.loc[mask_l5, "L5_TotalWeight"]
    )
    features.loc[mask_l5, "L5_Proj_U"] = (
        features.loc[mask_l5, "L5_sum_w_uU"] / features.loc[mask_l5, "L5_TotalWeight"]
    )

    # Cleanup temp columns
    features.drop(columns=[c for c in features.columns if "sum_w_u" in c], inplace=True)
    features.fillna(0.0, inplace=True)

    features = features.merge(df_epochs, on="utcTimeMillis", how="left")

    features = features.merge(imu_agg, on="utcTimeMillis", how="left")

    if gt_path:
        df_gt = pd.read_csv(os.path.join("./input", gt_path))
        df_gt = df_gt.rename(columns={"UnixTimeMillis": "utcTimeMillis"})

        features = features.merge(
            df_gt[
                [
                    "utcTimeMillis",
                    "LatitudeDegrees",
                    "LongitudeDegrees",
                    "AltitudeMeters",
                ]
            ],
            on="utcTimeMillis",
            how="inner",
            suffixes=("", "_GT"),
        )

        t_e, t_n, t_u = EcefToEnu.transform(
            *GeodeticToEcef.transform(
                features["LatitudeDegrees"].values,
                features["LongitudeDegrees"].values,
                features["AltitudeMeters"].values,
            ),
            features["Wls_Lat"].values,
            features["Wls_Lon"].values,
            features["Wls_Alt"].values,
        )

        features["Target_E"] = t_e
        features["Target_N"] = t_n
        features["Target_U"] = t_u

    features["tripId"] = f"{drive_id}-{phone_name}"
    features["drive_id"] = drive_id
    features["phone_name"] = phone_name

    return features


# Apply patch
GeometricFeatureExtractor._process_drive = patched_process_drive


if __name__ == "__main__":
    print("=== Smartphone Location Computation Demo ===")

    # Ensure working directory exists
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # 1. Load Metadata
    print("\n[1/7] Loading Metadata...")
    train_meta = load_metadata("train")
    test_meta = load_metadata("test")

    # Filter for a single drive to ensure speed
    target_drive = "2020-05-15-US-MTV-1"
    train_meta_subset = train_meta[train_meta["drive_id"] == target_drive].copy()

    target_test_trip = "2020-06-04-US-MTV-1-GooglePixel4"
    test_meta_subset = test_meta[test_meta["tripId"] == target_test_trip].copy()

    print(f"Training on drive: {target_drive} ({len(train_meta_subset)} rows)")
    print(f"Testing on trip: {target_test_trip} ({len(test_meta_subset)} rows)")

    # 2. Extract Features (Stream A)
    print("\n[2/7] Extracting Geometric Features...")
    feature_extractor = GeometricFeatureExtractor(cache_dir=CACHE_DIR)

    # Train Features
    train_features = feature_extractor.extract_features(
        train_meta_subset, load_cached_data=False
    )
    assert (
        train_features is not None and not train_features.empty
    ), "Train features extraction failed"

    # Test Features
    test_features = feature_extractor.extract_features(
        test_meta_subset, load_cached_data=False
    )
    assert (
        test_features is not None and not test_features.empty
    ), "Test features extraction failed"

    # 3. Train Residual Model
    print("\n[3/7] Training Residual Regressor...")
    model = ResidualRegressor(n_folds=2, model_dir=os.path.join(CACHE_DIR, "models"))

    # Optimize hyperparameters for speed
    model.params["n_estimators"] = 50
    model.params["num_leaves"] = 15

    model.train(train_features, load_cached_models=False)

    # 4. Predict Anchors
    print("\n[4/7] Predicting Anchors for Test Set...")
    anchors_df = model.predict(test_features)
    assert not anchors_df.empty, "Anchor prediction returned empty result"

    # 5. Compute Odometry (Stream B)
    print("\n[5/7] Computing Robust Odometry...")
    # run_odometry_processing writes to ./working/idea_17 internally
    odom_df = run_odometry_processing(test_meta_subset, load_cached_data=False)
    assert not odom_df.empty, "Odometry computation returned empty result"

    # 6. Trajectory Optimization
    print("\n[6/7] Running Factor Graph Optimization...")
    optimizer = GraphOptimizer(cache_dir=CACHE_DIR)

    # Align timestamps for optimizer (GeometricFeatureExtractor returns utcTimeMillis)
    if "utcTimeMillis" in anchors_df.columns:
        anchors_df = anchors_df.rename(columns={"utcTimeMillis": "UnixTimeMillis"})
    if "utcTimeMillis" in odom_df.columns:
        odom_df = odom_df.rename(columns={"utcTimeMillis": "UnixTimeMillis"})

    optimized_df = optimizer.run(anchors_df, odom_df, load_cached_data=False)

    # 7. Save Submission
    print("\n[7/7] Saving Submission...")
    save_submission(optimized_df, SUBMISSION_PATH)

    print("\n=== Demo Complete ===")
    print(f"Submission file created at: {SUBMISSION_PATH}")
