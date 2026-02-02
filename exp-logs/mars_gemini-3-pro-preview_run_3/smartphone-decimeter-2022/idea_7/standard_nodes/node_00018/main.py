import pandas as pd
import numpy as np
import random
import os
import sys

# Import from provided library files
from library.config import SEED
from library.feature_engineering import FeatureEngine
from library.model import ResidualBooster
from library.postprocessing import KinematicSmoother, generate_submission
from library.utils import calculate_score, haversine_distance


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup
    set_seed(SEED)
    print("Initializing Pipeline...")

    # 2. Feature Engineering (Train)
    # We must process train first to fit the encoders inside FeatureEngine
    engine = FeatureEngine()

    print("\n--- Processing Training Data ---")
    # Force reload to avoid stale cache (Cite debug_lesson_3)
    X_train, y_train, meta_train = engine.preprocess("train", load_cached_data=False)

    # Extract groups for Cross-Validation
    groups = meta_train["drive_id"]

    # Safety check to prevent ValueError if data is still small
    n_groups = groups.nunique()
    n_splits = 5
    if n_groups < n_splits:
        print(
            f"Warning: Number of groups ({n_groups}) is less than n_splits ({n_splits}). Adjusting n_splits."
        )
        n_splits = n_groups

    # 3. Model Training
    print("\n--- Training Model ---")
    booster = ResidualBooster()
    booster.train_cv(X_train, y_train, groups, n_splits=n_splits)

    # 4. Validation
    print("\n--- Processing Validation Data ---")
    X_val, y_val, meta_val = engine.preprocess("val", load_cached_data=False)

    print("Generating Validation Predictions...")
    pred_e_val, pred_n_val = booster.predict(X_val)

    # Prepare metadata for smoothing (needs predictions attached)
    meta_val_pred = meta_val.copy()
    meta_val_pred["pred_e"] = pred_e_val
    meta_val_pred["pred_n"] = pred_n_val

    # Apply Kinematic Smoothing
    print("Applying Kinematic Smoothing to Validation Set...")
    smoother = KinematicSmoother()
    val_smoothed = smoother.apply_smoothing(meta_val_pred)

    # Calculate Score
    # meta_val contains the Ground Truth (LatitudeDegrees, LongitudeDegrees)
    score = calculate_score(val_smoothed, meta_val)
    print(f"Final Validation Metric: {score}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude for each point
    val_errors = haversine_distance(
        val_smoothed["LatitudeDegrees"],
        val_smoothed["LongitudeDegrees"],
        meta_val["LatitudeDegrees"],
        meta_val["LongitudeDegrees"],
    )

    # Correlate errors with features
    analysis_df = X_val.copy()
    analysis_df["error_mag"] = val_errors
    correlations = analysis_df.corr()["error_mag"].sort_values(ascending=False)

    print("Correlation between Error Magnitude and Features:")
    print(correlations.drop("error_mag").head(10))

    # 5. Submission
    THRESHOLD = 4.32379283550646
    if score < THRESHOLD:
        print(
            f"\nValidation Score ({score}) meets threshold ({THRESHOLD}). Generating Submission..."
        )

        print("--- Processing Test Data ---")
        X_test, _, meta_test = engine.preprocess("test", load_cached_data=False)

        print("Generating Test Predictions...")
        pred_e_test, pred_n_test = booster.predict(X_test)

        generate_submission(meta_test, pred_e_test, pred_n_test)
        print("Pipeline Completed Successfully.")
    else:
        print(
            f"\nValidation Score ({score}) did not meet threshold ({THRESHOLD}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
