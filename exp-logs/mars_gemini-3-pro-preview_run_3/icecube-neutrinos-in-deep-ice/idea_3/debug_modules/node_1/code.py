import sys
import os
import numpy as np
import pandas as pd
import shutil
from pathlib import Path

# Ensure local library imports work
sys.path.append(os.getcwd())

import library.config as config
import library.utils as utils
import library.model
import library.trainer
from library.feature_engineering import FeatureExtractor
from library.model import DirectionalLGBM
from library.trainer import Trainer


def run_demo():
    print("Initializing Demo...")

    # ---------------------------------------------------------
    # 1. Setup & Configuration Overrides for Speed
    # ---------------------------------------------------------
    # Define fast hyperparameters for LightGBM to ensure quick execution
    FAST_LGBM_PARAMS = config.LGBM_PARAMS.copy()
    FAST_LGBM_PARAMS.update(
        {
            "num_leaves": 8,
            "max_depth": 3,
            "n_estimators": 10,
            "min_child_samples": 5,
            "verbose": -1,
        }
    )

    # Monkey-patch global constants in library modules to reduce training time
    library.model.NUM_BOOST_ROUND = 10
    library.model.EARLY_STOPPING_ROUNDS = 5

    # ---------------------------------------------------------
    # 2. Verify Utils
    # ---------------------------------------------------------
    print("\n--- Verifying Utils ---")

    # Test 1: Z-axis (Zenith=0) -> Cartesian (0, 0, 1)
    az, zen = 0.0, 0.0
    x, y, z = utils.spherical_to_cartesian(az, zen)
    assert (
        np.isclose(x, 0) and np.isclose(y, 0) and np.isclose(z, 1)
    ), f"Spherical to Cartesian failed for Zenith=0. Got ({x}, {y}, {z})"

    # Test 2: X-axis (Azimuth=0, Zenith=pi/2) -> Cartesian (1, 0, 0)
    az, zen = 0.0, np.pi / 2
    x, y, z = utils.spherical_to_cartesian(az, zen)
    assert (
        np.isclose(x, 1) and np.isclose(y, 0) and np.isclose(z, 0)
    ), f"Spherical to Cartesian failed for X-axis. Got ({x}, {y}, {z})"

    # Test 3: Round trip conversion
    az_in, zen_in = 1.5, 0.5
    x, y, z = utils.spherical_to_cartesian(az_in, zen_in)
    az_out, zen_out = utils.cartesian_to_spherical(x, y, z)
    assert np.isclose(az_in, az_out) and np.isclose(
        zen_in, zen_out
    ), "Round trip conversion failed."

    # Test 4: Angular Distance Metric
    # Distance between identical vectors should be 0
    dist = utils.angular_dist_score(
        np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0])
    )
    assert np.isclose(dist, 0.0), "Angular distance for identical vectors should be 0."

    # Distance between Z (zen=0) and -Z (zen=pi) should be pi
    dist = utils.angular_dist_score(0.0, 0.0, 0.0, np.pi)
    assert np.isclose(
        dist, np.pi
    ), f"Angular distance for opposite vectors should be pi. Got {dist}"

    print("Utils verification passed.")

    # ---------------------------------------------------------
    # 3. Verify Feature Engineering
    # ---------------------------------------------------------
    print("\n--- Verifying Feature Engineering ---")

    extractor = FeatureExtractor()

    # Load training metadata
    train_meta = pd.read_parquet(config.TRAIN_META_PATH)

    # Sample 50 events from the first batch found in metadata to minimize file I/O
    first_batch_id = train_meta["batch_id"].iloc[0]
    sample_meta = train_meta[train_meta["batch_id"] == first_batch_id].head(50).copy()

    print(
        f"Extracting features for {len(sample_meta)} events from batch {first_batch_id}..."
    )
    # Force re-computation (load_cached_data=False) to verify logic
    X, y, ids = extractor.extract_features(
        sample_meta, mode="train", load_cached_data=False
    )

    # Assertions
    assert len(X) == 50, f"Expected 50 feature rows, got {len(X)}"
    assert len(y) == 50, f"Expected 50 target rows, got {len(y)}"
    assert X.shape[1] == len(
        config.FEATURE_NAMES
    ), f"Feature count mismatch. Expected {len(config.FEATURE_NAMES)}, got {X.shape[1]}"
    assert y.shape[1] == 3, f"Target shape mismatch. Expected (N, 3), got {y.shape}"
    assert not np.isnan(X).any(), "Features contain NaNs."

    print("Feature Engineering verification passed.")

    # ---------------------------------------------------------
    # 4. Verify Model
    # ---------------------------------------------------------
    print("\n--- Verifying Model ---")

    # Initialize model with fast parameters
    model_wrapper = DirectionalLGBM(params=FAST_LGBM_PARAMS)

    # Split the extracted data for a quick train/val test
    split_idx = 40
    X_tr, y_tr = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]

    print("Training model on sample data...")
    model_wrapper.fit(X_tr, y_tr, X_val, y_val)

    print("Predicting...")
    pred_az, pred_zen = model_wrapper.predict(X_val)

    # Assertions
    assert len(pred_az) == 10, "Prediction length mismatch."
    assert np.all(
        (pred_az >= 0) & (pred_az <= 2 * np.pi + 1e-6)
    ), "Azimuth predictions out of range."
    assert np.all(
        (pred_zen >= 0) & (pred_zen <= np.pi + 1e-6)
    ), "Zenith predictions out of range."

    # Test Save/Load
    model_save_path = config.WORKING_DIR / "test_model.pkl"
    model_wrapper.save(model_save_path)
    assert model_save_path.exists(), "Model file was not saved."

    loaded_model = DirectionalLGBM(params=FAST_LGBM_PARAMS)
    loaded_model.load(model_save_path)
    assert len(loaded_model.models) == 3, "Loaded model does not contain 3 regressors."

    print("Model verification passed.")

    # ---------------------------------------------------------
    # 5. Verify Trainer (End-to-End)
    # ---------------------------------------------------------
    print("\n--- Verifying Trainer ---")

    trainer = Trainer()
    # Inject our fast model wrapper into the trainer
    trainer.model = DirectionalLGBM(params=FAST_LGBM_PARAMS)

    # 5.1 Train and Evaluate
    print("Running Trainer.train_and_evaluate...")
    # Using a small sample size to ensure speed
    trainer.train_and_evaluate(sample_size=100, load_cached_data=False)

    assert trainer.model_path.exists(), "Trainer failed to save the final model."

    # 5.2 Generate Submission
    # To avoid processing 13M test events, we create a dummy test metadata file
    print("Preparing dummy test metadata for submission verification...")

    real_test_meta = pd.read_parquet(config.TEST_META_PATH)
    # Pick events from one batch to ensure file existence
    test_batch_id = real_test_meta["batch_id"].iloc[0]
    dummy_test_meta = (
        real_test_meta[real_test_meta["batch_id"] == test_batch_id].head(20).copy()
    )

    dummy_test_meta_path = config.WORKING_DIR / "dummy_test_meta.parquet"
    dummy_test_meta.to_parquet(dummy_test_meta_path)

    # Monkey-patch the TEST_META_PATH in the library.trainer module to point to our dummy file
    library.trainer.TEST_META_PATH = dummy_test_meta_path

    print("Running Trainer.generate_submission...")
    trainer.generate_submission(load_cached_data=False)

    submission_path = config.SUBMISSION_DIR / "submission.csv"
    assert submission_path.exists(), "Submission file was not created."

    df_sub = pd.read_csv(submission_path)
    assert len(df_sub) == 20, f"Submission should have 20 rows, got {len(df_sub)}"
    assert list(df_sub.columns) == [
        "event_id",
        "azimuth",
        "zenith",
    ], "Submission columns mismatch."

    print("Trainer verification passed.")
    print("\nALL CHECKS PASSED.")


if __name__ == "__main__":
    # Ensure working directory exists
    config.WORKING_DIR.mkdir(parents=True, exist_ok=True)

    # Set fixed seeds
    np.random.seed(42)

    run_demo()
