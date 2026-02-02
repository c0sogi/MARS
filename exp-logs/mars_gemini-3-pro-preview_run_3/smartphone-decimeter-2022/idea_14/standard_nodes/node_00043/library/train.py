import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

from library.config import FEATURES, N_FOLDS, CACHE_DIR, SEED, SUBMISSION_PATH
from library.data_loader import get_train_data, get_val_data, get_test_data
from library.model import LGBMModel
from library.utils import ecef_to_geodetic, enu_to_ecef


def run_cross_validation(load_cached_data=True, n_folds=N_FOLDS, debug=False):
    """
    Orchestrates the cross-validation training workflow.
    Loads data, performs GroupKFold splitting, trains models, and saves artifacts.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.
        n_folds (int): Number of cross-validation folds.
        debug (bool): If True, subsamples the data for rapid testing.
    """
    # Ensure model directory exists
    model_dir = os.path.join(CACHE_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)

    print("Loading datasets...")
    # Load ONLY training data for Cross-Validation to avoid data leakage
    # Validation data is reserved for the final holdout evaluation in runfile.py
    # Cite debug_lesson_10: Ensuring data consistency by respecting split boundaries
    full_df = get_train_data(load_cached_data=load_cached_data)

    if debug:
        print("Debug mode: Subsampling data...")
        unique_drives = full_df["drive_id"].unique()
        sample_drives = unique_drives[:3] if len(unique_drives) > 3 else unique_drives
        full_df = full_df[full_df["drive_id"].isin(sample_drives)].reset_index(
            drop=True
        )

    print(f"Total samples for CV: {len(full_df)}")
    print(f"Unique drives: {full_df['drive_id'].nunique()}")

    # Initialize GroupKFold
    gkf = GroupKFold(n_splits=n_folds)
    groups = full_df["drive_id"]

    fold_scores_e = []
    fold_scores_n = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(full_df, groups=groups)):
        print(f"\n=== Fold {fold + 1}/{n_folds} ===")

        X_train_fold = full_df.iloc[train_idx].copy()
        X_val_fold = full_df.iloc[val_idx].copy()

        # Initialize and train model
        model = LGBMModel()

        # Train models for East and North residuals
        model.train(train_df=X_train_fold, val_df=X_val_fold, features=FEATURES)

        # Save models
        model_e_path = os.path.join(model_dir, f"lgbm_east_fold_{fold}.txt")
        model_n_path = os.path.join(model_dir, f"lgbm_north_fold_{fold}.txt")

        model.model_e.save_model(model_e_path)
        model.model_n.save_model(model_n_path)
        print(f"Saved models to {model_e_path} and {model_n_path}")

        # Evaluate
        pred_e, pred_n = model.predict(X_val_fold, FEATURES)

        mae_e = np.mean(np.abs(X_val_fold["target_E"] - pred_e))
        mae_n = np.mean(np.abs(X_val_fold["target_N"] - pred_n))

        print(f"Fold {fold + 1} MAE East: {mae_e}")
        print(f"Fold {fold + 1} MAE North: {mae_n}")

        fold_scores_e.append(mae_e)
        fold_scores_n.append(mae_n)

    print("\n=== Cross-Validation Results ===")
    print(f"Average MAE East: {np.mean(fold_scores_e)}")
    print(f"Average MAE North: {np.mean(fold_scores_n)}")
    print(
        f"Overall Average MAE: {(np.mean(fold_scores_e) + np.mean(fold_scores_n)) / 2}"
    )


def generate_submission(load_cached_data=True, n_folds=N_FOLDS):
    """
    Generates the submission file by averaging predictions from all fold models.

    Args:
        load_cached_data (bool): Whether to load pre-processed test data from cache.
        n_folds (int): Number of folds used in training (to load models).
    """
    print("\n=== Generating Submission ===")

    # Load Test Data
    test_df = get_test_data(load_cached_data=load_cached_data)
    print(f"Test data shape: {test_df.shape}")

    model_dir = os.path.join(CACHE_DIR, "models")

    # Containers for fold predictions
    preds_e_folds = []
    preds_n_folds = []

    # Load models and predict
    for fold in range(n_folds):
        model_e_path = os.path.join(model_dir, f"lgbm_east_fold_{fold}.txt")
        model_n_path = os.path.join(model_dir, f"lgbm_north_fold_{fold}.txt")

        if not os.path.exists(model_e_path) or not os.path.exists(model_n_path):
            print(f"Warning: Models for fold {fold} not found. Skipping.")
            continue

        bst_e = lgb.Booster(model_file=model_e_path)
        bst_n = lgb.Booster(model_file=model_n_path)

        preds_e_folds.append(bst_e.predict(test_df[FEATURES]))
        preds_n_folds.append(bst_n.predict(test_df[FEATURES]))

    if not preds_e_folds:
        raise RuntimeError("No models found to generate submission.")

    # Aggregate predictions (Pixel-wise Median for robustness)
    final_pred_e = np.median(np.column_stack(preds_e_folds), axis=1)
    final_pred_n = np.median(np.column_stack(preds_n_folds), axis=1)

    # Reconstruct Trajectory
    # 1. Get WLS Reference Coordinates
    wls_x = test_df["WlsPositionXEcefMeters"].values
    wls_y = test_df["WlsPositionYEcefMeters"].values
    wls_z = test_df["WlsPositionZEcefMeters"].values

    # 2. Get WLS Geodetic for rotation reference (Lat/Lon)
    # Note: We need WLS Altitude for accurate ENU->ECEF conversion
    wls_lat, wls_lon, wls_alt = ecef_to_geodetic(wls_x, wls_y, wls_z)

    # 3. Convert Predicted ENU Residuals to ECEF
    # Pred_E, Pred_N are offsets from WLS. We assume Up offset is 0.
    pred_x, pred_y, pred_z = enu_to_ecef(
        final_pred_e,
        final_pred_n,
        np.zeros_like(final_pred_e),  # Up residual = 0
        wls_lat,
        wls_lon,
        wls_alt,
    )

    # 4. Convert Corrected ECEF to Geodetic (Final Lat/Lon)
    pred_lat, pred_lon, _ = ecef_to_geodetic(pred_x, pred_y, pred_z)

    # Create Submission DataFrame
    submission = pd.DataFrame(
        {
            "tripId": test_df["tripId"],
            "UnixTimeMillis": test_df["UnixTimeMillis"],
            "LatitudeDegrees": pred_lat,
            "LongitudeDegrees": pred_lon,
        }
    )

    # Save
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
