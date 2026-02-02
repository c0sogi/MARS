import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from library.feature_engineering import process_data
from library.data_loader import load_dataset
from library.model import ResidualRegressor, apply_corrections
from library.utils import calculate_competition_metric

# Constants
METADATA_DIR = "./metadata"
SUBMISSION_DIR = "./submission"
WORKING_DIR = "./working/idea_12"
MODELS_DIR = os.path.join(WORKING_DIR, "models")


def save_boosters(model, output_dir):
    """
    Saves the trained LightGBM boosters from the ResidualRegressor ensemble to disk.

    Args:
        model (ResidualRegressor): The trained model object.
        output_dir (str): Directory to save the models.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save East models
    for i, booster in enumerate(model.models_e):
        path = os.path.join(output_dir, f"lgbm_east_fold_{i}.txt")
        booster.save_model(path)

    # Save North models
    for i, booster in enumerate(model.models_n):
        path = os.path.join(output_dir, f"lgbm_north_fold_{i}.txt")
        booster.save_model(path)

    print(
        f"Saved {len(model.models_e)} East models and {len(model.models_n)} North models to {output_dir}"
    )


def run_group_kfold(n_folds=5, load_cached_data=True, debug=False, seed=42):
    """
    Orchestrates the Group K-Fold cross-validation training process.

    Args:
        n_folds (int): Number of folds for cross-validation.
        load_cached_data (bool): Whether to load features from cache.
        debug (bool): If True, runs on a small subset of data.
        seed (int): Random seed for reproducibility.

    Returns:
        ResidualRegressor: The trained ensemble model.
    """
    print(f"Starting training pipeline (Debug={debug})...")

    # 1. Load and Process Training Data
    # process_data handles caching internally
    train_feats, train_targets = process_data(
        "train", load_cached_data=load_cached_data
    )

    # Load metadata to map tripId to drive_id for grouping
    # This ensures we don't leak data from the same drive into validation
    train_meta_path = os.path.join(METADATA_DIR, "train_metadata.csv")
    if not os.path.exists(train_meta_path):
        raise FileNotFoundError(f"Metadata not found at {train_meta_path}")

    train_meta = pd.read_csv(train_meta_path)
    trip_to_drive = dict(zip(train_meta["tripId"], train_meta["drive_id"]))

    # Align groups with features
    groups = train_feats["tripId"].map(trip_to_drive)

    # Prepare Feature Matrix X and Targets y
    drop_cols = ["tripId", "UnixTimeMillis"]
    X = train_feats.drop(columns=drop_cols)
    y_e = train_targets["target_E"]
    y_n = train_targets["target_N"]

    # Debug Mode: Subsample data
    if debug:
        print("Debug mode: Subsampling data...")
        sample_size = min(1000, len(X))
        X = X.iloc[:sample_size].copy()
        y_e = y_e.iloc[:sample_size].copy()
        y_n = y_n.iloc[:sample_size].copy()
        groups = groups.iloc[:sample_size].copy()
        train_feats = train_feats.iloc[
            :sample_size
        ].copy()  # Keep aligned for validation

    # 2. Train Ensemble
    model = ResidualRegressor(n_folds=n_folds, seed=seed)
    oof_e, oof_n = model.fit(X, y_e, y_n, groups)

    # Save trained models
    save_boosters(model, MODELS_DIR)

    # 3. Calculate Out-of-Fold Validation Score
    print("Calculating OOF Validation Score...")

    # We need WLS baselines to reconstruct the absolute positions from residuals
    # Load raw GNSS data (cached)
    train_gnss, _, train_gt = load_dataset("train", load_cached_data=load_cached_data)

    # Extract WLS positions
    wls_cols = [
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]

    # Group by timestamp to get unique WLS fix per epoch
    # Note: train_gnss might contain multiple rows per epoch (one per satellite),
    # but WLS position is repeated or we take the first one.
    wls_ref = (
        train_gnss.groupby(["tripId", "UnixTimeMillis"])[wls_cols].first().reset_index()
    )

    # Create Validation DataFrame aligned with OOF predictions
    val_df = train_feats[["tripId", "UnixTimeMillis"]].copy()

    # Merge WLS reference
    val_df = pd.merge(val_df, wls_ref, on=["tripId", "UnixTimeMillis"], how="left")

    # Apply corrections: WLS + Predicted Residuals -> Corrected LLA
    pred_lat, pred_lon = apply_corrections(val_df, oof_e, oof_n)

    val_df["LatitudeDegrees"] = pred_lat
    val_df["LongitudeDegrees"] = pred_lon

    # Calculate metric against Ground Truth
    # Ensure train_gt is filtered if debugging
    if debug:
        # We need to filter GT to match the sampled trips/times
        # Create a key for filtering
        val_keys = val_df[["tripId", "UnixTimeMillis"]]
        train_gt = pd.merge(
            val_keys, train_gt, on=["tripId", "UnixTimeMillis"], how="inner"
        )

    score = calculate_competition_metric(val_df, train_gt)

    # Print full precision as requested
    print(f"Validation Score: {score}")

    return model


def generate_submission(model, load_cached_data=True):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        model (ResidualRegressor): Trained model ensemble.
        load_cached_data (bool): Whether to load test features from cache.
    """
    print("Starting inference pipeline...")

    # 1. Load and Process Test Data
    test_feats, _ = process_data("test", load_cached_data=load_cached_data)

    drop_cols = ["tripId", "UnixTimeMillis"]
    X_test = test_feats.drop(columns=drop_cols)

    # 2. Predict Residuals
    print(f"Predicting on {len(X_test)} test samples...")
    pred_e, pred_n = model.predict(X_test)

    # 3. Apply Corrections to WLS Baseline
    print("Applying corrections to WLS baseline...")

    # Load Test GNSS for WLS
    test_gnss, _, _ = load_dataset("test", load_cached_data=load_cached_data)

    wls_cols = [
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]
    wls_ref_test = (
        test_gnss.groupby(["tripId", "UnixTimeMillis"])[wls_cols].first().reset_index()
    )

    # Prepare submission dataframe
    submission_df = test_feats[["tripId", "UnixTimeMillis"]].copy()

    # Merge WLS
    submission_df = pd.merge(
        submission_df, wls_ref_test, on=["tripId", "UnixTimeMillis"], how="left"
    )

    # Calculate final coordinates
    final_lat, final_lon = apply_corrections(submission_df, pred_e, pred_n)

    submission_df["LatitudeDegrees"] = final_lat
    submission_df["LongitudeDegrees"] = final_lon

    # 4. Save Submission
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    out_path = os.path.join(SUBMISSION_DIR, "submission.csv")

    out_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    submission_df[out_cols].to_csv(out_path, index=False)

    print(f"Submission saved to {out_path}")
    print(f"Submission shape: {submission_df.shape}")
