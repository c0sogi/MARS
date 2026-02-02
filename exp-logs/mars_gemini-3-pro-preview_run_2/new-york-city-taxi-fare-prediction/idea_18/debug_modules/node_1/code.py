import os
import sys
import shutil
import numpy as np
import pandas as pd
import xgboost as xgb

# ------------------------------------------------------------------------------
# 1. Configuration & Setup
# ------------------------------------------------------------------------------
# We import Config first to patch it for a fast demonstration run.
from library.config import Config

# Override Config for speed and isolation
print(">>> Configuring environment for demonstration...")
Config.LEARNER_SUBSAMPLE_SIZE = 50_000  # Reduce learner set to 50k rows for speed
Config.XGB_PARAMS["n_estimators"] = 20  # Reduce to 20 trees for quick training
Config.XGB_PARAMS["learning_rate"] = 0.1  # Higher LR for faster convergence in demo
Config.EARLY_STOPPING_ROUNDS = 5  # Quick early stopping
Config.WORKING_DIR = "./working/demo_execution"
Config.SUBMISSION_PATH = "./working/demo_outputs/submission.csv"
Config.VERBOSE_EVAL = 10  # Print eval every 10 rounds

# Clean up previous demo runs if any to ensure a fresh start
if os.path.exists(Config.WORKING_DIR):
    shutil.rmtree(Config.WORKING_DIR)
os.makedirs(Config.WORKING_DIR, exist_ok=True)
os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

# Import library modules after config patch
from library.utils import haversine_distance, compute_geohash_bins
from library.data_loader import load_training_data, load_validation_data, load_test_data
from library.feature_engineer import FeatureEngineer
from library.model_trainer import XGBTrainer


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def verify_utils():
    """Verifies correctness of utility functions."""
    print("\n>>> Verifying Utility Functions...")
    # Test Haversine: Distance between (0,0) and (0,1) degree
    # 1 degree lat is approx 111km.
    dist = haversine_distance(0, 0, 1, 0)
    assert np.isclose(
        dist, 111.19, atol=1.0
    ), f"Haversine calculation incorrect: {dist}"

    # Test Geohash Binning
    # Level 5 step is 0.045.
    # Point (BB_MIN_LAT + 0.05, BB_MIN_LON + 0.05) should be index 1,1 -> 10001
    # assuming Config.BB_MIN_LAT/LON are used as origin.
    lat = Config.BB_MIN_LAT + 0.05
    lon = Config.BB_MIN_LON + 0.05
    bin_id = compute_geohash_bins(np.array([lat]), np.array([lon]), level=5)[0]
    # lat_idx = floor(0.05 / 0.045) = 1
    # lon_idx = floor(0.05 / 0.045) = 1
    # id = 1 * 10000 + 1 = 10001
    assert bin_id == 10001, f"Geohash binning incorrect: {bin_id}"
    print("Utils verification passed.")


def main():
    set_seed(Config.RANDOM_SEED)

    # --------------------------------------------------------------------------
    # 2. Logic Verification
    # --------------------------------------------------------------------------
    verify_utils()

    # --------------------------------------------------------------------------
    # 3. Data Loading
    # --------------------------------------------------------------------------
    print("\n>>> Loading Data...")

    # Load Training Data (Wisdom + Learner)
    # load_training_data handles loading full parquet and splitting/subsampling learner
    # We force load_cached_data=False to demonstrate the full pipeline logic
    wisdom_df, learner_df = load_training_data(load_cached_data=False)

    print(f"Wisdom Set Shape (Original): {wisdom_df.shape}")
    print(f"Learner Set Shape (Subsampled): {learner_df.shape}")

    # Validate Data Loading
    assert not learner_df.empty, "Learner DF is empty"
    assert not wisdom_df.empty, "Wisdom DF is empty"
    assert (
        len(learner_df) <= Config.LEARNER_SUBSAMPLE_SIZE
    ), "Learner subsampling failed"

    # OPTIMIZATION: Downsample Wisdom DF for this demo to ensure stats computation is fast
    # The real pipeline uses full wisdom, but for demo we cap it at 100k
    if len(wisdom_df) > 100_000:
        print("Downsampling Wisdom Set to 100k for demonstration speed...")
        wisdom_df = wisdom_df.sample(n=100_000, random_state=Config.RANDOM_SEED)

    # Load Validation Data
    val_df = load_validation_data(load_cached_data=False)
    # Downsample Val for speed
    if len(val_df) > 20_000:
        val_df = val_df.sample(n=20_000, random_state=Config.RANDOM_SEED)
    print(f"Validation Set Shape: {val_df.shape}")

    # Load Test Data
    test_df = load_test_data(load_cached_data=False)
    print(f"Test Set Shape: {test_df.shape}")

    # --------------------------------------------------------------------------
    # 4. Feature Engineering
    # --------------------------------------------------------------------------
    print("\n>>> Running Feature Engineering...")
    fe = FeatureEngineer()

    # Process Train (Learner) - Uses K-Fold Subtraction
    # This fits the stats engine on wisdom_df and transforms learner_df
    processed_train = fe.process_train_data(
        wisdom_df, learner_df, load_cached_data=False
    )

    # Check for generated features
    expected_col = f"mean_fare_L{Config.GEOHASH_LEVELS[0]}"
    assert (
        expected_col in processed_train.columns
    ), f"Feature {expected_col} missing from train"
    assert "haversine_dist" in processed_train.columns, "Geometric features missing"
    # Check that we don't have all NaNs in stats columns
    assert (
        processed_train[expected_col].notna().sum() > 0
    ), "Statistical features are all NaN"

    # Process Validation - Uses Global Mapping
    processed_val = fe.process_validation_data(val_df, load_cached_data=False)
    assert (
        expected_col in processed_val.columns
    ), f"Feature {expected_col} missing from val"

    # Process Test - Uses Global Mapping
    processed_test = fe.process_test_data(test_df, load_cached_data=False)
    assert (
        expected_col in processed_test.columns
    ), f"Feature {expected_col} missing from test"

    print("Feature Engineering complete.")

    # --------------------------------------------------------------------------
    # 5. Model Training
    # --------------------------------------------------------------------------
    print("\n>>> Training Model...")
    trainer = XGBTrainer()

    # Verify params override
    assert (
        trainer.params["n_estimators"] == 20
    ), "Trainer did not pick up Config overrides"

    # Train
    trainer.fit(processed_train, processed_val)

    # Check feature importance to ensure model learned
    fi = trainer.get_feature_importance()
    print("\nTop 5 Features:")
    print(fi.head(5))
    assert not fi.empty, "Feature importance is empty"

    # --------------------------------------------------------------------------
    # 6. Evaluation & Submission
    # --------------------------------------------------------------------------
    print("\n>>> Evaluating and Generating Submission...")

    # Evaluate RMSE on Val
    rmse = trainer.evaluate(processed_val)
    print(f"Final Validation RMSE: {rmse:.4f}")
    assert rmse > 0, "RMSE should be positive"

    # Generate Submission
    trainer.generate_submission_file(processed_test)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert sub_df.shape == (
        len(test_df),
        2,
    ), f"Submission shape mismatch: {sub_df.shape}"
    assert list(sub_df.columns) == [
        "key",
        "fare_amount",
    ], "Submission columns incorrect"
    assert not sub_df.isnull().any().any(), "Submission contains NaNs"

    print("\n>>> Demonstration Run Complete Successfully.")


if __name__ == "__main__":
    main()
