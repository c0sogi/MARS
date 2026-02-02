import os
import pandas as pd
import numpy as np
import lightgbm as lgb

from library.config import FEATURES, N_FOLDS, CACHE_DIR, SUBMISSION_PATH
from library.data_loader import get_test_data
from library.utils import ecef_to_geodetic, enu_to_ecef


def generate_submission(load_cached_data=True, n_folds=N_FOLDS):
    """
    Generates the submission file by averaging predictions from all fold models.

    The pipeline follows these steps:
    1. Load the preprocessed test dataset (features + WLS baseline).
    2. Load trained LightGBM models for both East and North components for each fold.
    3. Generate predictions for the test set from each fold model.
    4. Aggregate predictions using pixel-wise median to handle outliers.
    5. Reconstruct the final trajectory by applying the predicted ENU residuals
       to the WLS baseline ECEF coordinates and converting to Geodetic (Lat/Lon).
    6. Save the results to the submission CSV file.

    Args:
        load_cached_data (bool): Whether to load pre-processed test data from cache.
        n_folds (int): Number of folds used in training (to load models).
    """
    print("\n=== Generating Submission ===")

    # 1. Load Test Data
    # The data loader handles caching and feature engineering
    test_df = get_test_data(load_cached_data=load_cached_data)
    print(f"Test data shape: {test_df.shape}")

    model_dir = os.path.join(CACHE_DIR, "models")
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"Model directory not found at {model_dir}. Please train models first."
        )

    # Containers for fold predictions
    # We will store predictions as (n_samples, n_folds) arrays
    preds_e_folds = []
    preds_n_folds = []

    # 2. & 3. Load models and predict
    print(f"Predicting with {n_folds} folds...")
    for fold in range(n_folds):
        model_e_path = os.path.join(model_dir, f"lgbm_east_fold_{fold}.txt")
        model_n_path = os.path.join(model_dir, f"lgbm_north_fold_{fold}.txt")

        if not os.path.exists(model_e_path) or not os.path.exists(model_n_path):
            print(f"Warning: Models for fold {fold} not found. Skipping this fold.")
            continue

        # Load LightGBM boosters
        bst_e = lgb.Booster(model_file=model_e_path)
        bst_n = lgb.Booster(model_file=model_n_path)

        # Predict
        # Note: LightGBM predict returns a numpy array
        pred_e = bst_e.predict(test_df[FEATURES])
        pred_n = bst_n.predict(test_df[FEATURES])

        preds_e_folds.append(pred_e)
        preds_n_folds.append(pred_n)

    if not preds_e_folds:
        raise RuntimeError("No models found to generate submission.")

    # 4. Aggregate predictions (Pixel-wise Median for robustness)
    # Stack arrays column-wise: (n_samples, n_folds)
    final_pred_e = np.median(np.column_stack(preds_e_folds), axis=1)
    final_pred_n = np.median(np.column_stack(preds_n_folds), axis=1)

    print("Predictions aggregated.")

    # 5. Reconstruct Trajectory
    print("Reconstructing trajectory from ENU residuals...")

    # Get WLS Reference Coordinates from the test dataframe
    # These are the baseline positions we are correcting
    wls_x = test_df["WlsPositionXEcefMeters"].values
    wls_y = test_df["WlsPositionYEcefMeters"].values
    wls_z = test_df["WlsPositionZEcefMeters"].values

    # Get WLS Geodetic coordinates for the rotation reference (Lat/Lon/Alt)
    # We need the WLS Altitude to accurately perform the ENU -> ECEF rotation
    wls_lat, wls_lon, wls_alt = ecef_to_geodetic(wls_x, wls_y, wls_z)

    # Convert Predicted ENU Residuals to ECEF offsets and apply to baseline
    # We assume the 'Up' residual is 0 since we only predict horizontal errors
    pred_u = np.zeros_like(final_pred_e)

    # enu_to_ecef returns the absolute ECEF coordinates (Baseline + Offset)
    pred_x, pred_y, pred_z = enu_to_ecef(
        final_pred_e,
        final_pred_n,
        pred_u,
        wls_lat,
        wls_lon,
        wls_alt,
    )

    # Convert Corrected ECEF to Geodetic (Final Latitude/Longitude)
    pred_lat, pred_lon, _ = ecef_to_geodetic(pred_x, pred_y, pred_z)

    # 6. Create Submission DataFrame
    submission = pd.DataFrame(
        {
            "tripId": test_df["tripId"],
            "UnixTimeMillis": test_df["UnixTimeMillis"],
            "LatitudeDegrees": pred_lat,
            "LongitudeDegrees": pred_lon,
        }
    )

    # Save submission file
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    print(f"Submission shape: {submission.shape}")
