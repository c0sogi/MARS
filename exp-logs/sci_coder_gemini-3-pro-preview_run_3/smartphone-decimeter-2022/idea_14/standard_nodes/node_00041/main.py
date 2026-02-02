import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Import Library Modules ---
# We need to modify config before importing other modules that might use it
import library.config as config

# Optimization: Reduce boosting rounds for faster execution within time limit
config.NUM_BOOST_ROUND = 1000
config.LGBM_PARAMS["verbose"] = -1

from library.train import run_cross_validation
from library.inference import generate_submission
from library.data_loader import get_val_data
from library.utils import ecef_to_geodetic, enu_to_ecef, haversine_distance
from library.config import FEATURES, N_FOLDS, CACHE_DIR


def evaluate_validation():
    """
    Evaluates the trained model ensemble on the validation dataset.
    Computes the competition metric and performs failure analysis.
    """
    print("\n=== Starting Validation Evaluation ===")

    # 1. Load Validation Data
    # load_cached_data=True allows using the parquet file if created by train.py
    val_df = get_val_data(load_cached_data=True)
    print(f"Validation data shape: {val_df.shape}")

    # 2. Load Models and Predict
    model_dir = os.path.join(CACHE_DIR, "models")
    preds_e_folds = []
    preds_n_folds = []

    print(f"Loading models from {model_dir} and predicting...")

    models_found = 0
    for fold in range(N_FOLDS):
        model_e_path = os.path.join(model_dir, f"lgbm_east_fold_{fold}.txt")
        model_n_path = os.path.join(model_dir, f"lgbm_north_fold_{fold}.txt")

        if not os.path.exists(model_e_path) or not os.path.exists(model_n_path):
            continue

        models_found += 1
        bst_e = lgb.Booster(model_file=model_e_path)
        bst_n = lgb.Booster(model_file=model_n_path)

        preds_e_folds.append(bst_e.predict(val_df[FEATURES]))
        preds_n_folds.append(bst_n.predict(val_df[FEATURES]))

    if models_found == 0:
        raise RuntimeError("No trained models found for validation!")

    # 3. Aggregate Predictions (Pixel-wise Median)
    pred_e_res = np.median(np.column_stack(preds_e_folds), axis=1)
    pred_n_res = np.median(np.column_stack(preds_n_folds), axis=1)

    # 4. Reconstruct Trajectory (ENU Residuals -> Geodetic)
    # Get WLS Reference
    wls_x = val_df["WlsPositionXEcefMeters"].values
    wls_y = val_df["WlsPositionYEcefMeters"].values
    wls_z = val_df["WlsPositionZEcefMeters"].values

    # Convert WLS ECEF to Geodetic (Lat/Lon/Alt) for rotation reference
    wls_lat, wls_lon, wls_alt = ecef_to_geodetic(wls_x, wls_y, wls_z)

    # Apply ENU offsets
    pred_x, pred_y, pred_z = enu_to_ecef(
        pred_e_res,
        pred_n_res,
        np.zeros_like(pred_e_res),  # Up residual = 0
        wls_lat,
        wls_lon,
        wls_alt,
    )

    # Convert back to Geodetic
    pred_lat, pred_lon, _ = ecef_to_geodetic(pred_x, pred_y, pred_z)

    # 5. Compute Metric
    # Calculate Haversine distance between Predicted and GT
    gt_lat = val_df["LatitudeDegrees"].values
    gt_lon = val_df["LongitudeDegrees"].values

    distances = haversine_distance(pred_lat, pred_lon, gt_lat, gt_lon)

    # Create temp dataframe for grouping
    score_df = pd.DataFrame({"tripId": val_df["tripId"], "error": distances})

    # Calculate 50th and 95th percentile per phone
    trip_scores = score_df.groupby("tripId")["error"].quantile([0.5, 0.95]).unstack()
    trip_scores["avg_metric"] = (trip_scores[0.5] + trip_scores[0.95]) / 2

    final_metric = trip_scores["avg_metric"].mean()

    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Correlate error magnitude with features
    analysis_df = val_df[FEATURES].copy()
    analysis_df["Error_Magnitude"] = distances

    correlations = analysis_df.corr()["Error_Magnitude"].sort_values(ascending=False)
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations.drop("Error_Magnitude"))

    return final_metric


def main():
    # 1. Train Models
    print("Starting Training Pipeline...")
    # We use the provided training orchestration
    run_cross_validation(load_cached_data=False, n_folds=N_FOLDS)

    # 2. Evaluate on Validation Set
    val_metric = evaluate_validation()

    # 3. Generate Submission if metric is good
    THRESHOLD = 4.2637075691068755
    if val_metric < THRESHOLD:
        print(
            f"\nValidation metric {val_metric} < {THRESHOLD}. Generating submission..."
        )
        generate_submission(load_cached_data=False, n_folds=N_FOLDS)
    else:
        print(
            f"\nValidation metric {val_metric} >= {THRESHOLD}. Submission generation skipped."
        )


if __name__ == "__main__":
    main()
