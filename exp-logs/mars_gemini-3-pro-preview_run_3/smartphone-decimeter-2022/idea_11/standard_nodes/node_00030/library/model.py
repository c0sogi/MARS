import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import warnings

from library.config import (
    LGBM_PARAMS,
    N_FOLDS,
    SEED,
    WORKING_DIR,
    FEATURE_COLS,
    TARGET_COLS,
    SUBMISSION_PATH,
    EARLY_STOPPING_ROUNDS,
    VERBOSE_EVAL,
    WGS84_A,
    WGS84_F,
    WGS84_B,
)
from library.data_loader import load_train_data, load_val_data, load_test_data
from library.features import ecef_to_lla

# Suppress warnings
warnings.filterwarnings("ignore")


class PhysicsEnsembleModel:
    def __init__(self):
        self.models_east = []
        self.models_north = []
        self.feature_cols = FEATURE_COLS

    def _get_radii(self, lat_deg):
        """
        Calculate Meridian (M) and Prime Vertical (N) radii of curvature.
        """
        lat_rad = np.radians(lat_deg)
        a = WGS84_A
        # e^2 = 2f - f^2
        e2 = 2 * WGS84_F - WGS84_F**2

        sin_lat = np.sin(lat_rad)
        tmp = 1 - e2 * sin_lat**2

        # Prime Vertical Radius
        N = a / np.sqrt(tmp)

        # Meridian Radius
        M = a * (1 - e2) / (tmp**1.5)

        return M, N

    def reconstruct_coords(self, df, pred_east, pred_north):
        """
        Convert WLS ECEF to LLA, then apply ENU residuals to get final Lat/Lon.
        """
        # 1. Convert WLS ECEF to LLA
        wls_x = df["WlsPositionXEcefMeters"].values
        wls_y = df["WlsPositionYEcefMeters"].values
        wls_z = df["WlsPositionZEcefMeters"].values

        wls_lat, wls_lon, _ = ecef_to_lla(wls_x, wls_y, wls_z)

        # 2. Convert ENU residuals to Lat/Lon offsets
        # dLat = dN / M
        # dLon = dE / (N * cos(lat))

        M, N = self._get_radii(wls_lat)

        d_lat_rad = pred_north / M
        d_lon_rad = pred_east / (N * np.cos(np.radians(wls_lat)))

        pred_lat = wls_lat + np.degrees(d_lat_rad)
        pred_lon = wls_lon + np.degrees(d_lon_rad)

        return pred_lat, pred_lon

    def train(self, train_df, val_df=None):
        """
        Train the ensemble using GroupKFold.
        Combines train and val sets for CV to maximize data usage,
        or uses them as is if specific validation strategy is preferred.
        Here we combine them and use drive_id groups.
        """
        print("Preparing data for training...")

        # Combine datasets if val is provided
        if val_df is not None:
            full_df = pd.concat([train_df, val_df], ignore_index=True)
        else:
            full_df = train_df

        X = full_df[self.feature_cols]
        y_east = full_df["delta_east"]
        y_north = full_df["delta_north"]
        groups = full_df["drive_id"]

        # Initialize GroupKFold
        gkf = GroupKFold(n_splits=N_FOLDS)

        print(f"Starting training with {N_FOLDS} folds...")

        fold_metrics = {"east": [], "north": []}

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_east, groups)):
            print(f"\n--- Fold {fold + 1} ---")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_east_train, y_east_val = y_east.iloc[train_idx], y_east.iloc[val_idx]
            y_north_train, y_north_val = y_north.iloc[train_idx], y_north.iloc[val_idx]

            # --- Train East Model ---
            print("Training East Model...")
            dtrain_e = lgb.Dataset(X_train, label=y_east_train)
            dval_e = lgb.Dataset(X_val, label=y_east_val, reference=dtrain_e)

            model_e = lgb.train(
                LGBM_PARAMS,
                dtrain_e,
                valid_sets=[dtrain_e, dval_e],
                valid_names=["train", "val"],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
                    lgb.log_evaluation(VERBOSE_EVAL),
                ],
            )

            # Save model
            model_path_e = os.path.join(WORKING_DIR, f"lgbm_east_fold_{fold}.txt")
            model_e.save_model(model_path_e)
            self.models_east.append(model_e)

            # Evaluate
            pred_e = model_e.predict(X_val, num_iteration=model_e.best_iteration)
            mae_e = mean_absolute_error(y_east_val, pred_e)
            fold_metrics["east"].append(mae_e)
            print(f"Fold {fold + 1} East MAE: {mae_e}")

            # --- Train North Model ---
            print("Training North Model...")
            dtrain_n = lgb.Dataset(X_train, label=y_north_train)
            dval_n = lgb.Dataset(X_val, label=y_north_val, reference=dtrain_n)

            model_n = lgb.train(
                LGBM_PARAMS,
                dtrain_n,
                valid_sets=[dtrain_n, dval_n],
                valid_names=["train", "val"],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
                    lgb.log_evaluation(VERBOSE_EVAL),
                ],
            )

            # Save model
            model_path_n = os.path.join(WORKING_DIR, f"lgbm_north_fold_{fold}.txt")
            model_n.save_model(model_path_n)
            self.models_north.append(model_n)

            # Evaluate
            pred_n = model_n.predict(X_val, num_iteration=model_n.best_iteration)
            mae_n = mean_absolute_error(y_north_val, pred_n)
            fold_metrics["north"].append(mae_n)
            print(f"Fold {fold + 1} North MAE: {mae_n}")

            # Combined metric for this fold (mean of 50/95 not calc here, just avg MAE)
            print(f"Fold {fold + 1} Average MAE: {(mae_e + mae_n) / 2}")

        print("\n=== Training Completed ===")
        print(f"Average East MAE: {np.mean(fold_metrics['east'])}")
        print(f"Average North MAE: {np.mean(fold_metrics['north'])}")
        print(
            f"Overall MAE: {(np.mean(fold_metrics['east']) + np.mean(fold_metrics['north'])) / 2}"
        )

    def predict(self, test_df):
        """
        Generate predictions using the ensemble.
        Aggregates predictions using pixel-wise median.
        """
        print("Generating predictions...")
        X_test = test_df[self.feature_cols]

        # Predict East
        preds_east = []
        for model in self.models_east:
            preds_east.append(model.predict(X_test, num_iteration=model.best_iteration))

        # Predict North
        preds_north = []
        for model in self.models_north:
            preds_north.append(
                model.predict(X_test, num_iteration=model.best_iteration)
            )

        # Stack and Compute Median
        # Shape: (n_folds, n_samples)
        stack_east = np.vstack(preds_east)
        stack_north = np.vstack(preds_north)

        # Median aggregation to filter outliers
        median_east = np.median(stack_east, axis=0)
        median_north = np.median(stack_north, axis=0)

        return median_east, median_north

    def run_pipeline(self):
        """
        Orchestrates the full pipeline: Load -> Train -> Predict -> Save.
        """
        # 1. Load Data
        print("Loading datasets...")
        train_df = load_train_data(load_cached_data=True)
        val_df = load_val_data(load_cached_data=True)
        test_df = load_test_data(load_cached_data=True)

        # 2. Train
        self.train(train_df, val_df)

        # 3. Predict
        pred_east, pred_north = self.predict(test_df)

        # 4. Reconstruct Coordinates
        print("Reconstructing coordinates...")
        pred_lat, pred_lon = self.reconstruct_coords(test_df, pred_east, pred_north)

        # 5. Create Submission
        print(f"Saving submission to {SUBMISSION_PATH}...")
        submission = pd.DataFrame(
            {
                "tripId": test_df["tripId"],
                "UnixTimeMillis": test_df["UnixTimeMillis"],
                "LatitudeDegrees": pred_lat,
                "LongitudeDegrees": pred_lon,
            }
        )

        submission.to_csv(SUBMISSION_PATH, index=False)
        print("Submission saved successfully.")


def train_ensemble_model():
    """
    Wrapper function to instantiate and run the model pipeline.
    """
    model = PhysicsEnsembleModel()
    model.run_pipeline()
