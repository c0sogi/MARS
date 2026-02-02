import os
import glob
import pandas as pd
import numpy as np
import lightgbm as lgb
from library.feature_engineering import process_data
from library.data_loader import load_dataset
from library.model import ResidualRegressor, apply_corrections
from library.utils import ecef_to_wgs84

# Constants
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
WORKING_DIR = "./working/idea_12"
MODELS_DIR = os.path.join(WORKING_DIR, "models")


def load_trained_models(models_dir):
    """
    Loads trained LightGBM models from the specified directory into a ResidualRegressor instance.

    Args:
        models_dir (str): Path to the directory containing model files.

    Returns:
        ResidualRegressor: An instance with loaded boosters.
    """
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    model = ResidualRegressor()

    # Load East models
    east_files = sorted(glob.glob(os.path.join(models_dir, "lgbm_east_fold_*.txt")))
    if not east_files:
        print("Warning: No East models found in directory.")

    for f in east_files:
        # Load booster from text file
        booster = lgb.Booster(model_file=f)
        model.models_e.append(booster)

    # Load North models
    north_files = sorted(glob.glob(os.path.join(models_dir, "lgbm_north_fold_*.txt")))
    if not north_files:
        print("Warning: No North models found in directory.")

    for f in north_files:
        # Load booster from text file
        booster = lgb.Booster(model_file=f)
        model.models_n.append(booster)

    print(
        f"Loaded {len(model.models_e)} East models and {len(model.models_n)} North models."
    )
    return model


def generate_submission(load_cached_data=True):
    """
    Generates predictions for the test set using the ensemble of trained models
    and saves the final submission file.

    Args:
        load_cached_data (bool): Whether to use cached features/data if available.
    """
    print("Starting submission generation pipeline...")

    # 1. Load Trained Models
    print(f"Loading models from {MODELS_DIR}...")
    model = load_trained_models(MODELS_DIR)

    if not model.models_e or not model.models_n:
        raise RuntimeError(
            "Models not loaded correctly. Cannot proceed with inference."
        )

    # 2. Load and Process Test Data
    # process_data handles caching of features internally
    print("Loading and processing test features...")
    test_feats, _ = process_data("test", load_cached_data=load_cached_data)

    # Prepare feature matrix
    drop_cols = ["tripId", "UnixTimeMillis"]
    X_test = test_feats.drop(columns=drop_cols)

    # 3. Predict Residuals
    # ResidualRegressor.predict implements the Pixel-wise Median aggregation
    print(f"Predicting residuals for {len(X_test)} samples...")
    pred_e, pred_n = model.predict(X_test)

    # 4. Load WLS Baseline for Reconstruction
    print("Loading WLS baseline from test GNSS data...")
    # load_dataset handles caching of raw sensor data
    test_gnss, _, _ = load_dataset("test", load_cached_data=load_cached_data)

    # Extract WLS positions columns
    wls_cols = [
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]

    # Check if WLS columns exist
    if not all(c in test_gnss.columns for c in wls_cols):
        raise ValueError("WLS columns missing in test GNSS data.")

    # Group by tripId and timestamp to get unique WLS fix per epoch
    # We take the first entry as WLS is constant for the epoch
    wls_ref_test = (
        test_gnss.groupby(["tripId", "UnixTimeMillis"])[wls_cols].first().reset_index()
    )

    # 5. Apply Corrections
    print("Applying ENU corrections to WLS baseline...")

    # Prepare submission dataframe structure aligned with features
    submission_df = test_feats[["tripId", "UnixTimeMillis"]].copy()

    # Merge WLS baseline into submission dataframe
    submission_df = pd.merge(
        submission_df, wls_ref_test, on=["tripId", "UnixTimeMillis"], how="left"
    )

    # Calculate corrected Lat/Lon using the utility function
    final_lat, final_lon = apply_corrections(submission_df, pred_e, pred_n)

    submission_df["LatitudeDegrees"] = final_lat
    submission_df["LongitudeDegrees"] = final_lon

    # Handle any potential NaNs (fallback to WLS)
    if submission_df[["LatitudeDegrees", "LongitudeDegrees"]].isnull().any().any():
        print(
            "Warning: NaNs detected in predictions. Falling back to WLS baseline for missing values."
        )
        wls_lat, wls_lon, _ = ecef_to_wgs84(
            submission_df["WlsPositionXEcefMeters"].values,
            submission_df["WlsPositionYEcefMeters"].values,
            submission_df["WlsPositionZEcefMeters"].values,
        )
        submission_df["LatitudeDegrees"] = submission_df["LatitudeDegrees"].fillna(
            pd.Series(wls_lat)
        )
        submission_df["LongitudeDegrees"] = submission_df["LongitudeDegrees"].fillna(
            pd.Series(wls_lon)
        )

    # 6. Save Submission
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Format according to competition requirements
    out_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    submission_df[out_cols].to_csv(SUBMISSION_PATH, index=False)

    print(f"Submission saved to {SUBMISSION_PATH}")
    print(f"Submission shape: {submission_df.shape}")
