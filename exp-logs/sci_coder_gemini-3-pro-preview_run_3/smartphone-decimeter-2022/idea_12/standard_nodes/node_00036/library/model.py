import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from library.utils import (
    ecef_to_wgs84,
    enu_to_ecef,
    wgs84_to_ecef,
    calculate_competition_metric,
)
from library.feature_engineering import process_data
from library.data_loader import load_dataset

# Constants
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_12"
SEED = 42

# Ensure directories exist
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(WORKING_DIR, exist_ok=True)


class ResidualRegressor:
    """
    Ensemble of LightGBM regressors predicting ENU residuals.
    Uses MAE (L1) objective for robustness.
    """

    def __init__(self, n_folds=5, seed=SEED):
        self.n_folds = n_folds
        self.seed = seed
        self.models_e = []  # List of models for East component
        self.models_n = []  # List of models for North component
        self.feature_cols = []

    def fit(self, X, y_e, y_n, groups):
        """
        Trains the ensemble using GroupKFold.

        Args:
            X (pd.DataFrame): Feature matrix.
            y_e (pd.Series): Target East residuals.
            y_n (pd.Series): Target North residuals.
            groups (pd.Series): Group labels (drive_id) for CV.
        """
        self.feature_cols = X.columns.tolist()
        self.models_e = []
        self.models_n = []

        gkf = GroupKFold(n_splits=self.n_folds)

        # LightGBM Parameters
        params = {
            "objective": "regression_l1",  # MAE for robustness
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbosity": -1,
            "seed": self.seed,
            "n_jobs": -1,
        }

        print(f"Training {self.n_folds}-fold ensemble...")

        oof_preds_e = np.zeros(len(X))
        oof_preds_n = np.zeros(len(X))

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_e, groups)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            ye_train, ye_val = y_e.iloc[train_idx], y_e.iloc[val_idx]
            yn_train, yn_val = y_n.iloc[train_idx], y_n.iloc[val_idx]

            # Train East Model
            dtrain_e = lgb.Dataset(X_train, label=ye_train)
            dval_e = lgb.Dataset(X_val, label=ye_val, reference=dtrain_e)

            callbacks = [
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),  # Disable logging
            ]

            model_e = lgb.train(
                params,
                dtrain_e,
                num_boost_round=2000,
                valid_sets=[dval_e],
                callbacks=callbacks,
            )
            self.models_e.append(model_e)
            oof_preds_e[val_idx] = model_e.predict(
                X_val, num_iteration=model_e.best_iteration
            )

            # Train North Model
            dtrain_n = lgb.Dataset(X_train, label=yn_train)
            dval_n = lgb.Dataset(X_val, label=yn_val, reference=dtrain_n)

            model_n = lgb.train(
                params,
                dtrain_n,
                num_boost_round=2000,
                valid_sets=[dval_n],
                callbacks=callbacks,
            )
            self.models_n.append(model_n)
            oof_preds_n[val_idx] = model_n.predict(
                X_val, num_iteration=model_n.best_iteration
            )

        # Calculate OOF MAE
        mae_e = np.mean(np.abs(y_e - oof_preds_e))
        mae_n = np.mean(np.abs(y_n - oof_preds_n))
        print(f"OOF MAE - East: {mae_e:.4f}, North: {mae_n:.4f}")

        return oof_preds_e, oof_preds_n

    def predict(self, X):
        """
        Predicts residuals using the ensemble.
        Aggregates predictions using pixel-wise median.

        Args:
            X (pd.DataFrame): Feature matrix.

        Returns:
            tuple: (pred_e, pred_n)
        """
        if not self.models_e or not self.models_n:
            raise RuntimeError("Models not trained.")

        # Collect predictions from all folds
        preds_e_all = []
        preds_n_all = []

        for model in self.models_e:
            preds_e_all.append(model.predict(X, num_iteration=model.best_iteration))

        for model in self.models_n:
            preds_n_all.append(model.predict(X, num_iteration=model.best_iteration))

        # Pixel-wise Median Aggregation
        pred_e = np.median(np.column_stack(preds_e_all), axis=1)
        pred_n = np.median(np.column_stack(preds_n_all), axis=1)

        return pred_e, pred_n


def apply_corrections(wls_df, pred_e, pred_n):
    """
    Applies ENU residuals to WLS baseline to get corrected Lat/Lon.

    Args:
        wls_df: DataFrame with WlsPosition[X/Y/Z]EcefMeters
        pred_e: Predicted East residual (meters)
        pred_n: Predicted North residual (meters)

    Returns:
        tuple: (lat_corrected, lon_corrected)
    """
    # Extract WLS ECEF
    wls_x = wls_df["WlsPositionXEcefMeters"].values
    wls_y = wls_df["WlsPositionYEcefMeters"].values
    wls_z = wls_df["WlsPositionZEcefMeters"].values

    # Convert WLS ECEF to LLA (Reference Point)
    ref_lat, ref_lon, ref_alt = ecef_to_wgs84(wls_x, wls_y, wls_z)

    # Convert Predicted ENU Residuals to ECEF Offsets
    # We assume Delta Up is 0 for horizontal correction
    pred_u = np.zeros_like(pred_e)

    # Note: enu_to_ecef returns the Absolute ECEF coordinates, not offsets
    # The function signature is enu_to_ecef(e, n, u, lat0, lon0, alt0) -> x, y, z
    # Since pred_e/n are residuals relative to WLS (GT - WLS),
    # adding them to WLS in ENU frame gives the corrected position.

    corr_x, corr_y, corr_z = enu_to_ecef(
        pred_e, pred_n, pred_u, ref_lat, ref_lon, ref_alt
    )

    # Convert Corrected ECEF to LLA
    lat_pred, lon_pred, _ = ecef_to_wgs84(corr_x, corr_y, corr_z)

    return lat_pred, lon_pred


def main(load_cached_data=True):
    print("Starting Sector-Aware Physics-Boosted Residual Ensemble Pipeline...")

    # -------------------------------------------------------------------------
    # 1. Load and Process Training Data
    # -------------------------------------------------------------------------
    print("\n--- Processing Training Data ---")
    # This loads features and computes targets (ENU residuals)
    train_feats, train_targets = process_data(
        "train", load_cached_data=load_cached_data
    )

    # Load metadata to get drive_id for grouping
    train_meta = pd.read_csv(os.path.join(METADATA_DIR, "train_metadata.csv"))

    # Create mapping tripId -> drive_id
    trip_to_drive = dict(zip(train_meta["tripId"], train_meta["drive_id"]))

    # Map groups
    # Ensure alignment: train_feats and train_targets are aligned by process_data
    groups = train_feats["tripId"].map(trip_to_drive)

    # Drop non-feature columns
    drop_cols = ["tripId", "UnixTimeMillis"]
    X_train = train_feats.drop(columns=drop_cols)
    y_train_e = train_targets["target_E"]
    y_train_n = train_targets["target_N"]

    # -------------------------------------------------------------------------
    # 2. Train Model
    # -------------------------------------------------------------------------
    print("\n--- Training Model ---")
    model = ResidualRegressor(n_folds=5, seed=SEED)
    oof_e, oof_n = model.fit(X_train, y_train_e, y_train_n, groups)

    # -------------------------------------------------------------------------
    # 3. Validation Scoring (Optional but recommended)
    # -------------------------------------------------------------------------
    print("\n--- Calculating Validation Score ---")
    # To calculate the competition metric, we need to apply OOF residuals to WLS
    # and compare with Ground Truth.

    # Load raw GNSS to get WLS for training set (needed for OOF reconstruction)
    # We use load_dataset which caches.
    train_gnss, _, train_gt = load_dataset("train", load_cached_data=True)

    # We need to align WLS with the features/predictions
    # train_feats has unique tripId, UnixTimeMillis
    # train_gnss has multiple rows. We group to get WLS.
    wls_cols = [
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]

    # Ensure train_gnss has WLS columns
    if all(c in train_gnss.columns for c in wls_cols):
        # Group to get 1 row per timestamp
        wls_ref = (
            train_gnss.groupby(["tripId", "UnixTimeMillis"])[wls_cols]
            .first()
            .reset_index()
        )

        # Merge with prediction index (train_feats) to ensure alignment
        # train_feats matches X_train order
        val_df = train_feats[["tripId", "UnixTimeMillis"]].copy()
        val_df = pd.merge(val_df, wls_ref, on=["tripId", "UnixTimeMillis"], how="left")

        # Apply OOF corrections
        pred_lat, pred_lon = apply_corrections(val_df, oof_e, oof_n)

        val_df["LatitudeDegrees"] = pred_lat
        val_df["LongitudeDegrees"] = pred_lon

        # Calculate Score
        score = calculate_competition_metric(val_df, train_gt)
        print(f"Validation Competition Score (OOF): {score}")
    else:
        print(
            "Warning: WLS columns missing in training data. Skipping validation score calculation."
        )

    # -------------------------------------------------------------------------
    # 4. Process Test Data
    # -------------------------------------------------------------------------
    print("\n--- Processing Test Data ---")
    test_feats, _ = process_data("test", load_cached_data=load_cached_data)

    X_test = test_feats.drop(columns=drop_cols)

    # -------------------------------------------------------------------------
    # 5. Inference
    # -------------------------------------------------------------------------
    print("\n--- Inference ---")
    pred_e, pred_n = model.predict(X_test)

    # -------------------------------------------------------------------------
    # 6. Apply Corrections and Save
    # -------------------------------------------------------------------------
    print("\n--- Generating Submission ---")
    # Load Test GNSS for WLS baseline
    test_gnss, _, _ = load_dataset("test", load_cached_data=load_cached_data)

    # Get WLS for test set
    wls_ref_test = (
        test_gnss.groupby(["tripId", "UnixTimeMillis"])[wls_cols].first().reset_index()
    )

    # Merge with test features to align
    submission_df = test_feats[["tripId", "UnixTimeMillis"]].copy()
    submission_df = pd.merge(
        submission_df, wls_ref_test, on=["tripId", "UnixTimeMillis"], how="left"
    )

    # Apply predictions
    final_lat, final_lon = apply_corrections(submission_df, pred_e, pred_n)

    submission_df["LatitudeDegrees"] = final_lat
    submission_df["LongitudeDegrees"] = final_lon

    # Select required columns
    out_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    final_submission = submission_df[out_cols]

    # Save
    final_submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    print(f"Submission shape: {final_submission.shape}")


if __name__ == "__main__":
    main()
