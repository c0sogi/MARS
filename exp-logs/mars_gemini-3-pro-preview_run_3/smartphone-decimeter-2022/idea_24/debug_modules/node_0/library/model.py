import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import GroupKFold
from library.config import Config
from library.utils import llh_to_ecef, ecef_to_llh, ecef_to_enu, enu_to_ecef
from library.data_io import load_metadata, load_drive_data
from library.feature_engineering import compute_geometric_features

# Ensure model directory exists
MODEL_DIR = os.path.join(Config.WORKING_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


class LightGBMRegressor:
    """
    Wrapper for LightGBM training and prediction with GroupKFold support.
    """

    def __init__(self, target_name):
        self.target_name = target_name
        self.models = []
        self.feature_names = []

    def train_group_kfold(self, X, y, groups, n_splits=5):
        """
        Trains LightGBM models using GroupKFold cross-validation.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.
            groups (pd.Series): Group labels (drive_id) for splitting.
            n_splits (int): Number of folds.
        """
        self.feature_names = X.columns.tolist()
        self.models = []

        gkf = GroupKFold(n_splits=n_splits)

        print(f"\nTraining {self.target_name} model with {n_splits} folds...")

        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            # Create LightGBM datasets
            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

            # Train model
            model = lgb.train(
                Config.LGBM_PARAMS,
                dtrain,
                valid_sets=[dtrain, dval],
                valid_names=["train", "val"],
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=Config.EARLY_STOPPING_ROUNDS, verbose=False
                    ),
                    lgb.log_evaluation(period=0),  # Silent training
                ],
            )

            # Evaluate
            y_pred = model.predict(X_val, num_iteration=model.best_iteration)
            score = np.mean(np.abs(y_val - y_pred))  # MAE
            fold_scores.append(score)

            print(f"Fold {fold+1} MAE: {score:.8f}")

            self.models.append(model)

            # Save model
            model_path = os.path.join(
                MODEL_DIR, f"lgbm_{self.target_name}_fold_{fold}.txt"
            )
            model.save_model(model_path)

        print(f"Average MAE for {self.target_name}: {np.mean(fold_scores):.8f}")

    def predict(self, X):
        """
        Generates predictions by averaging outputs from all fold models.

        Args:
            X (pd.DataFrame): Feature matrix.

        Returns:
            np.ndarray: Averaged predictions.
        """
        if not self.models:
            raise ValueError("Models not trained yet.")

        preds = np.zeros(len(X))
        for model in self.models:
            preds += model.predict(X, num_iteration=model.best_iteration)

        return preds / len(self.models)


def prepare_dataset(split="train", load_cached_data=True):
    """
    Loads, aligns, and computes features/targets for a dataset split.
    Combines 'train' and 'val' metadata if split='train' to maximize training data.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to use cached intermediate files.

    Returns:
        pd.DataFrame: The processed dataset with features and targets (if available).
    """
    # Define cache path for the aligned dataset
    cache_path = Config.get_cache_path(f"dataset_{split}")

    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading {split} dataset from cache: {cache_path}")
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print(f"Preparing {split} dataset from source...")

    # Load metadata
    if split == "train":
        # Combine train and val metadata for full training
        meta_train = load_metadata("train")
        meta_val = load_metadata("val")
        meta_df = pd.concat([meta_train, meta_val], ignore_index=True)
    else:
        meta_df = load_metadata(split)

    # Limit for debugging
    if Config.DEBUG:
        meta_df = meta_df.sample(
            n=min(len(meta_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).copy()

    # Get unique drives
    unique_drives = meta_df[["drive_id", "phone_name"]].drop_duplicates()

    dfs = []

    for _, row in unique_drives.iterrows():
        drive_id = row["drive_id"]
        phone_name = row["phone_name"]

        # Get subset of metadata for this drive
        drive_meta = meta_df[
            (meta_df["drive_id"] == drive_id) & (meta_df["phone_name"] == phone_name)
        ].copy()
        if drive_meta.empty:
            continue

        # 1. Compute Geometric Features
        # Get paths from the first row of this drive
        sample_row = drive_meta.iloc[0]
        gnss_path = sample_row["gnss_path"]
        imu_path = sample_row["imu_path"]

        features = compute_geometric_features(
            drive_id, phone_name, gnss_path, imu_path, load_cached_data=load_cached_data
        )

        if features.empty:
            continue

        # 2. Load WLS Positions (from GNSS file)
        data = load_drive_data(
            drive_id, phone_name, gnss_path, imu_path, load_cached_data=load_cached_data
        )
        gnss_df = data["gnss"]

        # We need WLS position for every timestamp to calculate ENU targets/residuals
        # Columns: utcTimeMillis, WlsPositionXEcefMeters, ...
        wls_cols = [
            "utcTimeMillis",
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
        ]
        if not all(c in gnss_df.columns for c in wls_cols):
            continue

        wls_df = gnss_df[wls_cols].dropna().drop_duplicates(subset=["utcTimeMillis"])
        wls_df = wls_df.rename(columns={"utcTimeMillis": "UnixTimeMillis"})

        # 3. Merge Features, WLS, and Metadata (GT)
        # Metadata contains the target GT Lat/Lon and timestamps
        merged = pd.merge(drive_meta, features, on="UnixTimeMillis", how="inner")
        merged = pd.merge(merged, wls_df, on="UnixTimeMillis", how="inner")

        # 4. Compute Targets (ENU Residuals) if training
        if split != "test":
            # Convert GT (Lat, Lon, Alt) to ECEF
            # Metadata usually only has Lat/Lon. We use WLS Alt as approximation for GT Alt
            # to compute horizontal error, or 0. Using WLS Alt is safer for local linearization.
            # However, ground_truth.csv usually has AltitudeMeters.
            # If loaded via load_drive_data('gt'), we would have it.
            # But meta_df only has Lat/Lon.
            # Let's load full GT to get Altitude if possible, otherwise use WLS Alt.

            # Actually, let's use the WLS position as the reference for ENU conversion.
            # Target = GT_pos - WLS_pos (in ENU frame centered at WLS_pos)

            # Convert WLS ECEF to LLH to get reference Lat/Lon/Alt
            wls_x = merged["WlsPositionXEcefMeters"].values
            wls_y = merged["WlsPositionYEcefMeters"].values
            wls_z = merged["WlsPositionZEcefMeters"].values

            # Vectorized conversion
            # We iterate or use a vectorized implementation.
            # Since utils functions are scalar, we wrap them or use simple approximation.
            # For speed in this script, we'll iterate (dataset size is manageable after aggregation).

            targets_e = []
            targets_n = []

            gt_lats = merged["LatitudeDegrees"].values
            gt_lons = merged["LongitudeDegrees"].values

            for i in range(len(merged)):
                wx, wy, wz = wls_x[i], wls_y[i], wls_z[i]
                gt_lat, gt_lon = gt_lats[i], gt_lons[i]

                # Get WLS LLH as reference
                ref_lat, ref_lon, ref_alt = ecef_to_llh(wx, wy, wz)

                # Convert GT to ECEF (using ref_alt to project onto same height surface)
                gt_x, gt_y, gt_z = llh_to_ecef(gt_lat, gt_lon, ref_alt)

                # Convert GT ECEF to ENU relative to WLS
                e, n, u = ecef_to_enu(gt_x, gt_y, gt_z, ref_lat, ref_lon, ref_alt)

                targets_e.append(e)
                targets_n.append(n)

            merged["Target_E"] = targets_e
            merged["Target_N"] = targets_n

        dfs.append(merged)

    if not dfs:
        return pd.DataFrame()

    full_df = pd.concat(dfs, ignore_index=True)

    # Save to cache
    try:
        full_df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save dataset cache: {e}")

    return full_df


def train_models(load_cached_data=True):
    """
    Trains the LightGBM models for East and North residuals.
    """
    # 1. Prepare Data
    df_train = prepare_dataset("train", load_cached_data=load_cached_data)

    if df_train.empty:
        print("Error: No training data available.")
        return

    # 2. Define Features
    # Exclude metadata and targets
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

    feature_cols = [c for c in df_train.columns if c not in exclude_cols]
    print(f"Training with {len(feature_cols)} features: {feature_cols}")

    X = df_train[feature_cols]
    groups = df_train["drive_id"]

    # 3. Train East Model
    model_e = LightGBMRegressor("Target_E")
    model_e.train_group_kfold(X, df_train["Target_E"], groups)

    # 4. Train North Model
    model_n = LightGBMRegressor("Target_N")
    model_n.train_group_kfold(X, df_train["Target_N"], groups)

    print("Training complete.")


def generate_submission(load_cached_data=True):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    # 1. Prepare Test Data
    df_test = prepare_dataset("test", load_cached_data=load_cached_data)

    if df_test.empty:
        print("Error: No test data available.")
        return

    # 2. Load Models
    # We need to reload the models trained in train_models
    # Since LightGBMRegressor stores list of boosters, we need to reconstruct it or load from files
    # For simplicity, we assume models are saved in MODEL_DIR with standard names

    # Identify feature columns
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
    feature_cols = [c for c in df_test.columns if c not in exclude_cols]
    X_test = df_test[feature_cols]

    # Helper to load and predict
    def load_and_predict(target_name):
        preds = np.zeros(len(X_test))
        fold = 0
        model_count = 0
        while True:
            model_path = os.path.join(MODEL_DIR, f"lgbm_{target_name}_fold_{fold}.txt")
            if not os.path.exists(model_path):
                break

            model = lgb.Booster(model_file=model_path)
            preds += model.predict(X_test)
            fold += 1
            model_count += 1

        if model_count > 0:
            return preds / model_count
        else:
            print(f"Warning: No models found for {target_name}")
            return np.zeros(len(X_test))

    pred_e = load_and_predict("Target_E")
    pred_n = load_and_predict("Target_N")

    # 3. Reconstruct Absolute Positions
    # WLS + Pred_ENU -> Pred_LLH

    wls_x = df_test["WlsPositionXEcefMeters"].values
    wls_y = df_test["WlsPositionYEcefMeters"].values
    wls_z = df_test["WlsPositionZEcefMeters"].values

    pred_lats = []
    pred_lons = []

    for i in range(len(df_test)):
        wx, wy, wz = wls_x[i], wls_y[i], wls_z[i]
        de, dn = pred_e[i], pred_n[i]

        # Reference WLS LLH
        ref_lat, ref_lon, ref_alt = ecef_to_llh(wx, wy, wz)

        # Convert predicted ENU offset to ECEF
        # Assume dU = 0 for 2D correction
        dx, dy, dz = enu_to_ecef(de, dn, 0, ref_lat, ref_lon, ref_alt)

        # New ECEF
        # Note: enu_to_ecef returns absolute coords, not delta
        # Wait, check utils.py implementation of enu_to_ecef
        # It returns x, y, z absolute. Correct.

        # Convert back to LLH
        plat, plon, _ = ecef_to_llh(dx, dy, dz)

        pred_lats.append(plat)
        pred_lons.append(plon)

    # 4. Save Submission
    submission = pd.DataFrame(
        {
            "tripId": df_test["tripId"],
            "UnixTimeMillis": df_test["UnixTimeMillis"],
            "LatitudeDegrees": pred_lats,
            "LongitudeDegrees": pred_lons,
        }
    )

    output_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
