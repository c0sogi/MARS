import os
import numpy as np
import pandas as pd
import warnings

# Import library components
from library.config import Config
from library.feature_engineering import get_processed_dataset
from library.model_interface import train_residual_model, apply_correction
from library.data_loader import load_drive_data
from library.carrier_phase import get_tdcp_displacement
from library.trajectory_optimizer import TrajectoryAligner
from library.evaluation import score_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("Starting Demonstration Script...")

    # -------------------------------------------------------------------------
    # 1. Configuration & Patching for Speed
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment...")
    # Enable debug mode to potentially limit internal data sampling if implemented
    Config.DEBUG = True

    # Reduce LightGBM estimators for fast demonstration
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["min_child_samples"] = 5  # Reduce to allow split on small data

    # Patch NUM_FOLDS to match the small data sample
    Config.NUM_FOLDS = 2  # Fix: Ensure splits <= groups (Cite debug_lesson_8)

    # Ensure reproducibility
    np.random.seed(Config.SEED)

    print("Configuration patched for speed (NUM_FOLDS=2).")

    # -------------------------------------------------------------------------
    # 2. Data Loading & Feature Engineering
    # -------------------------------------------------------------------------
    print("\n[2] Loading and processing data...")

    # Load a very small subset of drives for training and validation
    # We use max_drives=2 to ensure we have enough data to run but not wait long
    train_df = get_processed_dataset(
        split="train", load_cached_data=False, max_drives=2
    )
    val_df = get_processed_dataset(split="val", load_cached_data=False, max_drives=1)

    print(f"Training Data Shape: {train_df.shape}")
    print(f"Validation Data Shape: {val_df.shape}")

    assert not train_df.empty, "Training dataframe is empty!"
    assert not val_df.empty, "Validation dataframe is empty!"

    # Verify key columns exist
    required_cols = (
        Config.FEATURE_COLUMNS + Config.TARGET_COLUMNS + ["Wls_X", "Wls_Y", "Wls_Z"]
    )
    for col in required_cols:
        assert col in train_df.columns, f"Missing column {col} in training data"

    # -------------------------------------------------------------------------
    # 3. Model Training
    # -------------------------------------------------------------------------
    print("\n[3] Training Residual Regressor...")

    # Train model (force_retrain is handled inside via load_cached_model=False)
    regressor = train_residual_model(train_df, load_cached_model=False)

    print("Model training complete.")

    # -------------------------------------------------------------------------
    # 4. Inference & Correction
    # -------------------------------------------------------------------------
    print("\n[4] Running Inference on Validation Set...")

    # Predict residuals
    pred_E, pred_N = regressor.predict(val_df)

    # Apply corrections to WLS baseline to get Lat/Lon
    val_df["LatitudeDegrees_Pred"], val_df["LongitudeDegrees_Pred"] = apply_correction(
        val_df, pred_E, pred_N
    )

    # Create a prediction dataframe formatted for evaluation/optimization
    # Note: We need UnixTimeMillis for merging
    val_preds = val_df[
        [
            "tripId",
            "UnixTimeMillis",
            "drive_id",
            "phone_name",
            "LatitudeDegrees_Pred",
            "LongitudeDegrees_Pred",
        ]
    ].rename(
        columns={
            "LatitudeDegrees_Pred": "LatitudeDegrees",
            "LongitudeDegrees_Pred": "LongitudeDegrees",
        }
    )

    print("Inference complete. Sample predictions:")
    print(val_preds.head(3))

    # -------------------------------------------------------------------------
    # 5. Trajectory Optimization (TDCP)
    # -------------------------------------------------------------------------
    print("\n[5] Running Trajectory Optimization (TDCP) on a single drive...")

    # Pick the first drive from validation set
    sample_drive_id = val_preds.iloc[0]["drive_id"]
    sample_phone_name = val_preds.iloc[0]["phone_name"]
    trip_id = val_preds.iloc[0]["tripId"]

    print(f"Optimizing Drive: {sample_drive_id}, Phone: {sample_phone_name}")

    # Load raw GNSS for this drive to compute TDCP
    df_gnss, _, _ = load_drive_data(sample_drive_id, sample_phone_name)
    assert not df_gnss.empty, "Failed to load raw GNSS data for optimization"

    # Compute TDCP displacements
    # load_cached_data=False to ensure we run the logic
    tdcp_df = get_tdcp_displacement(
        sample_drive_id, sample_phone_name, df_gnss, load_cached_data=False
    )
    print(f"TDCP Data Shape: {tdcp_df.shape}")

    # Filter predictions for this specific trip
    drive_preds = val_preds[val_preds["tripId"] == trip_id].copy()

    # Run Optimization
    aligner = TrajectoryAligner()
    # We patch the huber delta or lambda if needed, but defaults are fine

    optimized_df = aligner.optimize_drive(
        sample_drive_id, sample_phone_name, drive_preds, tdcp_df, load_cached_data=False
    )

    print("Optimization complete.")
    print(f"Optimized Data Shape: {optimized_df.shape}")

    assert len(optimized_df) == len(drive_preds), "Optimized output length mismatch"

    # -------------------------------------------------------------------------
    # 6. Evaluation
    # -------------------------------------------------------------------------
    print("\n[6] Evaluating Results...")

    # Get Ground Truth for this drive
    # We can extract it from val_df which contains the GT columns
    gt_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    drive_gt = val_df[val_df["tripId"] == trip_id][gt_cols].copy()

    # Score ML Baseline
    score_ml = score_submission(drive_preds, drive_gt)
    print(f"ML Baseline Score (Mean 50/95 Error): {score_ml:.4f} meters")

    # Score Optimized
    score_opt = score_submission(optimized_df, drive_gt)
    print(f"Optimized Score (Mean 50/95 Error):  {score_opt:.4f} meters")

    # Basic sanity check: Optimization shouldn't explode the error
    # (It might not always improve it on a tiny subset with un-tuned params, but shouldn't be massive)
    if not np.isnan(score_opt):
        assert (
            score_opt < 1000.0
        ), "Optimization produced wildly incorrect results (>1km error)"

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
