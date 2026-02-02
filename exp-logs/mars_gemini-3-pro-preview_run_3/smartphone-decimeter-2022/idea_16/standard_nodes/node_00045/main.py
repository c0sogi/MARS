import os
import sys
import numpy as np
import pandas as pd
import random
import torch
import warnings

# Import from provided library files
from library.config import Config
from library.data_loader import load_dataset, load_drive_data
from library.model_interface import train_residual_model, apply_correction
from library.carrier_phase import get_tdcp_displacement
from library.trajectory_optimizer import TrajectoryAligner
from library.evaluation import score_submission, calculate_distance_errors

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Set fixed random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_trajectory_optimization(df_features, mode="val"):
    """
    Apply Carrier-Phase Trajectory Alignment (TDCP + Optimization) to the ML predictions.

    Args:
        df_features (pd.DataFrame): Dataframe containing ML predictions and identifiers.
        mode (str): 'val' or 'test' to determine data loading source.

    Returns:
        pd.DataFrame: Dataframe with optimized LatitudeDegrees and LongitudeDegrees.
    """
    print(f"Running Trajectory Optimization for {mode} set...")

    aligner = TrajectoryAligner()
    optimized_rows = []

    # Group by drive to process trajectories sequentially
    groups = df_features.groupby(["drive_id", "phone_name"])

    for (drive_id, phone_name), group in groups:
        # 1. Load Raw GNSS for TDCP
        # Note: We rely on caching inside get_tdcp_displacement to make this fast on re-runs
        df_gnss, _, _ = load_drive_data(drive_id, phone_name)

        if df_gnss is None or df_gnss.empty:
            # Fallback to ML predictions if raw data missing
            print(
                f"Warning: No raw GNSS found for {drive_id}-{phone_name}. Skipping optimization."
            )
            optimized_rows.append(
                group[
                    ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
                ]
            )
            continue

        # 2. Compute TDCP Displacements
        df_tdcp = get_tdcp_displacement(drive_id, phone_name, df_gnss)

        # 3. Optimize Trajectory
        # The group df contains the ML predictions in LatitudeDegrees/LongitudeDegrees
        df_opt = aligner.optimize_drive(drive_id, phone_name, group, df_tdcp)

        optimized_rows.append(df_opt)

    # Concatenate all optimized trajectories
    if optimized_rows:
        result_df = pd.concat(optimized_rows, ignore_index=True)
    else:
        result_df = pd.DataFrame(
            columns=["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
        )

    return result_df


def main():
    print("Starting End-to-End Pipeline...")
    set_seed(Config.SEED)

    # -------------------------------------------------------------------------
    # 1. Training
    # -------------------------------------------------------------------------
    print("\n=== Phase 1: Model Training ===")
    # Load processed training data (cached if available)
    train_df = load_dataset("train", load_cached_data=True)

    # Subsample if necessary for speed (though 200k rows is small for LGBM)
    if len(train_df) > 500000:
        print(f"Subsampling training data from {len(train_df)} to 500000...")
        train_df = train_df.sample(n=500000, random_state=Config.SEED)

    # Train Residual Regressor
    model = train_residual_model(
        train_df, load_cached_model=False
    )  # Force retrain for baseline validity

    # -------------------------------------------------------------------------
    # 2. Validation & Optimization
    # -------------------------------------------------------------------------
    print("\n=== Phase 2: Validation & Optimization ===")
    val_df = load_dataset("val", load_cached_data=True)

    # Predict Residuals using ML
    pred_E, pred_N = model.predict(val_df)

    # Apply corrections to WLS baseline to get ML-only coordinates
    ml_lat, ml_lon = apply_correction(val_df, pred_E, pred_N)

    # Create a dataframe for optimization input
    val_pred_df = val_df[["tripId", "UnixTimeMillis", "drive_id", "phone_name"]].copy()
    val_pred_df["LatitudeDegrees"] = ml_lat
    val_pred_df["LongitudeDegrees"] = ml_lon

    # Run Trajectory Optimization (TDCP)
    val_opt_df = run_trajectory_optimization(val_pred_df, mode="val")

    # -------------------------------------------------------------------------
    # 3. Evaluation
    # -------------------------------------------------------------------------
    print("\n=== Phase 3: Evaluation ===")

    # The validation dataframe from load_dataset contains the GT columns (LatitudeDegrees, LongitudeDegrees)
    # We need to rename them to avoid confusion or extract them for scoring
    gt_df = val_df[
        ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    ].copy()

    # Score the Optimized Predictions
    final_metric = score_submission(val_opt_df, gt_df)

    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 4. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n=== Phase 4: Failure Analysis ===")

    # Merge predictions with GT to calculate errors
    analysis_df = pd.merge(
        val_opt_df, gt_df, on=["tripId", "UnixTimeMillis"], suffixes=("_pred", "_gt")
    )

    # Calculate distance error
    analysis_df["error_meters"] = calculate_distance_errors(
        analysis_df,
        "LatitudeDegrees_gt",
        "LongitudeDegrees_gt",
        "LatitudeDegrees_pred",
        "LongitudeDegrees_pred",
    )

    # Merge back features from val_df for correlation analysis
    # We use UnixTimeMillis and tripId as keys
    feature_analysis_df = pd.merge(analysis_df, val_df, on=["tripId", "UnixTimeMillis"])

    # Compute correlations
    print("Correlation between Error Magnitude and Features:")
    correlations = {}
    for col in Config.FEATURE_COLUMNS:
        if col in feature_analysis_df.columns:
            corr = feature_analysis_df["error_meters"].corr(feature_analysis_df[col])
            correlations[col] = corr

    # Sort and print
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, corr in sorted_corrs[:10]:
        print(f"{feat}: {corr:.4f}")

    # -------------------------------------------------------------------------
    # 5. Inference & Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 4.202107392205921

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating Submission..."
        )

        # Load Test Data
        test_df = load_dataset("test", load_cached_data=True)

        # Predict Residuals
        t_pred_E, t_pred_N = model.predict(test_df)

        # Apply ML Corrections
        t_lat, t_lon = apply_correction(test_df, t_pred_E, t_pred_N)

        # Prepare for Optimization
        test_pred_df = test_df[
            ["tripId", "UnixTimeMillis", "drive_id", "phone_name"]
        ].copy()
        test_pred_df["LatitudeDegrees"] = t_lat
        test_pred_df["LongitudeDegrees"] = t_lon

        # Run Trajectory Optimization
        test_opt_df = run_trajectory_optimization(test_pred_df, mode="test")

        # Format Submission
        submission_df = test_opt_df[
            ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
        ]

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
