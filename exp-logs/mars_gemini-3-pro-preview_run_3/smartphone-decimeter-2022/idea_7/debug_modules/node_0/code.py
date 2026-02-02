import os
import sys
import numpy as np
import pandas as pd
import importlib

# 1. CONFIGURATION OVERRIDE
# We modify the config module directly to enable DEBUG mode and reduce training time
# before importing dependent modules.
import library.config

library.config.DEBUG = True
library.config.DEBUG_DRIVE_COUNT = 3  # Use a small subset of drives
library.config.NUM_BOOST_ROUND = 10  # Very few rounds for demonstration speed
library.config.EARLY_STOPPING_ROUNDS = 5

# Reload modules to ensure they pick up the modified config variables
import library.data_loader

importlib.reload(library.data_loader)
import library.feature_engineering

importlib.reload(library.feature_engineering)
import library.model

importlib.reload(library.model)
import library.postprocessing

importlib.reload(library.postprocessing)

from library.feature_engineering import FeatureEngine
from library.model import ResidualBooster
from library.postprocessing import KinematicSmoother
from library.utils import calculate_score


def main():
    print("=== Starting Pipeline Demonstration ===")

    # -------------------------------------------------------------------------
    # 2. FEATURE ENGINEERING
    # -------------------------------------------------------------------------
    print("\n[Step 1] Feature Engineering...")
    engine = FeatureEngine()

    # Process Training Data
    # load_cached_data=False forces re-computation to demonstrate the logic
    print("Processing Training Split...")
    X_train, y_train, meta_train = engine.preprocess("train", load_cached_data=False)

    # Process Validation Data
    print("Processing Validation Split...")
    X_val, y_val, meta_val = engine.preprocess("val", load_cached_data=False)

    # Assertions to verify data integrity
    assert not X_train.empty, "X_train should not be empty"
    assert not y_train.empty, "y_train should not be empty"
    assert len(X_train) == len(y_train), "Mismatch in Train X and y length"
    assert (
        "phone_idx" in X_train.columns
    ), "Feature engineering failed to encode phone_name"

    print(f"Train Data Shape: {X_train.shape}")
    print(f"Val Data Shape:   {X_val.shape}")

    # -------------------------------------------------------------------------
    # 3. MODEL TRAINING
    # -------------------------------------------------------------------------
    print("\n[Step 2] Model Training...")
    booster = ResidualBooster()

    # We use 'drive_id' from metadata as groups for GroupKFold
    # Note: meta_train might have multiple rows per drive, but drive_id is consistent
    groups = meta_train[
        "tripId"
    ]  # Using tripId as group for stricter splitting in this demo

    # Train the model
    booster.train_cv(X_train, y_train, groups=groups, n_splits=3)

    # Verify models were saved/created
    assert len(booster.models_east) > 0, "East models not trained"
    assert len(booster.models_north) > 0, "North models not trained"

    # -------------------------------------------------------------------------
    # 4. INFERENCE
    # -------------------------------------------------------------------------
    print("\n[Step 3] Inference on Validation Set...")
    pred_e, pred_n = booster.predict(X_val)

    assert len(pred_e) == len(X_val), "Prediction length mismatch"

    # Attach predictions to validation metadata for post-processing
    val_results = meta_val.copy()
    val_results["pred_e"] = pred_e
    val_results["pred_n"] = pred_n

    # Calculate Baseline Score (WLS only) vs Raw Prediction Score
    # Note: Target was (GT - WLS), so Predicted Position = WLS + Pred
    val_results["Lat_RawPred"] = val_results[
        "wls_lat"
    ]  # Approximation for scoring check logic
    # Real conversion happens in post-processing or manually here.
    # Let's skip manual conversion and rely on the Smoother to do it properly.

    # -------------------------------------------------------------------------
    # 5. POST-PROCESSING (SMOOTHING)
    # -------------------------------------------------------------------------
    print("\n[Step 4] Kinematic Smoothing...")
    smoother = KinematicSmoother(
        innovation_threshold=50.0
    )  # Relaxed threshold for demo

    # apply_smoothing expects columns: tripId, UnixTimeMillis, wls_e, wls_n, pred_e, pred_n, wls_u, anchors...
    # These are all present in 'val_results' (meta_val + preds)
    smoothed_df = smoother.apply_smoothing(val_results)

    assert (
        "LatitudeDegrees" in smoothed_df.columns
    ), "Smoother failed to produce Latitude"
    assert len(smoothed_df) == len(val_results), "Smoother dropped rows"

    # -------------------------------------------------------------------------
    # 6. EVALUATION
    # -------------------------------------------------------------------------
    print("\n[Step 5] Evaluation...")

    # Prepare Ground Truth DataFrame for scoring
    # meta_val contains the GT columns 'LatitudeDegrees' and 'LongitudeDegrees'
    gt_df = meta_val[
        ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    ].copy()

    # Prepare Prediction DataFrame
    # smoothed_df contains the smoothed 'LatitudeDegrees' and 'LongitudeDegrees'
    pred_df = smoothed_df[
        ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    ].copy()

    # Calculate Score
    score = calculate_score(pred_df, gt_df)

    print(f"\nFinal Validation Score (Mean 50/95 Percentile Error): {score:.4f} meters")

    # Basic sanity check on score
    if np.isnan(score):
        print("Warning: Score is NaN. Check data overlap.")
    else:
        assert score > 0, "Score must be positive"

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
