import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)

# Import library modules
# We import config first to modify hyperparameters before they are used by other modules
import library.config as config
import library.utils as utils
from library.feature_engineering import FeatureEngineeringPipeline
from library.model import VectorRegressor


def main():
    print("=== Starting Neutrino Direction Prediction Demo ===\n")

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    print("Step 1: Configuring environment for fast demonstration...")

    # Modify global config for speed
    # We reduce the number of estimators and leaves to make training instant
    config.LGBM_PARAMS["n_estimators"] = 10
    config.LGBM_PARAMS["num_leaves"] = 15
    config.LGBM_PARAMS["max_depth"] = 3
    config.LGBM_PARAMS["min_child_samples"] = 5

    # Define a small debug size for data loading
    DEBUG_SIZE = 500

    # Ensure working directories exist (handled by config, but good practice)
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    print(f"  Debug size set to: {DEBUG_SIZE}")
    print(f"  LGBM n_estimators set to: {config.LGBM_PARAMS['n_estimators']}")

    # ---------------------------------------------------------
    # 2. Verify Utility Functions
    # ---------------------------------------------------------
    print("\nStep 2: Verifying utility logic (Coordinate Transforms)...")

    # Test Case 1: Zenith pointing straight up (z-axis)
    # Azimuth=Any, Zenith=0 -> x=0, y=0, z=1
    az_test, zen_test = 0.0, 0.0
    x, y, z = utils.spherical_to_cartesian(az_test, zen_test)
    assert (
        np.isclose(x, 0) and np.isclose(y, 0) and np.isclose(z, 1)
    ), f"Failed Zenith=0 check: Got ({x}, {y}, {z})"

    # Test Case 2: Pointing along Y-axis
    # Azimuth=pi/2, Zenith=pi/2 -> x=0, y=1, z=0
    az_test, zen_test = np.pi / 2, np.pi / 2
    x, y, z = utils.spherical_to_cartesian(az_test, zen_test)
    assert (
        np.isclose(x, 0, atol=1e-6)
        and np.isclose(y, 1, atol=1e-6)
        and np.isclose(z, 0, atol=1e-6)
    ), f"Failed Y-axis check: Got ({x}, {y}, {z})"

    # Test Case 3: Round Trip
    x_in, y_in, z_in = 1.0, 0.0, 0.0  # X-axis
    az_out, zen_out = utils.cartesian_to_spherical(x_in, y_in, z_in)
    # Expect Az=0, Zen=pi/2
    assert np.isclose(az_out, 0) and np.isclose(
        zen_out, np.pi / 2
    ), f"Failed Round Trip check: Got Az={az_out}, Zen={zen_out}"

    # Test Case 4: Angular Error
    # Angle between X-axis (1,0,0) and Y-axis (0,1,0) should be pi/2
    az1, zen1 = 0, np.pi / 2
    az2, zen2 = np.pi / 2, np.pi / 2
    error = utils.compute_angular_error(
        np.array([az1]), np.array([zen1]), np.array([az2]), np.array([zen2])
    )
    assert np.isclose(
        error, np.pi / 2
    ), f"Failed Angular Error check: Got {error}, expected {np.pi/2}"

    print("  All utility assertions passed.")

    # ---------------------------------------------------------
    # 3. Data Loading (Feature Engineering)
    # ---------------------------------------------------------
    print("\nStep 3: Loading Data (Debug Mode)...")

    # Initialize pipeline with debug size
    pipeline = FeatureEngineeringPipeline(debug_size=DEBUG_SIZE)

    # Load data
    # We set load_cached_data=False to force the loader to process the raw files
    # and respect the new DEBUG_SIZE, rather than loading a potentially larger existing cache.
    print("  Loading Training set...")
    X_train, y_train, ids_train = pipeline.get_train_data(load_cached_data=False)

    print("  Loading Validation set...")
    X_val, y_val, ids_val = pipeline.get_val_data(load_cached_data=False)

    print("  Loading Test set...")
    X_test, _, ids_test = pipeline.get_test_data(load_cached_data=False)

    # Assertions on data shape
    print(f"  Train Shape: {X_train.shape}, Targets: {y_train.shape}")
    print(f"  Val Shape:   {X_val.shape}, Targets: {y_val.shape}")
    print(f"  Test Shape:  {X_test.shape}")

    assert (
        len(X_train) == DEBUG_SIZE
    ), f"Train size mismatch. Expected {DEBUG_SIZE}, got {len(X_train)}"
    assert y_train.shape[1] == 3, "Target should have 3 columns (x, y, z)"
    assert not np.isnan(X_train).any(), "Training data contains NaNs"

    # ---------------------------------------------------------
    # 4. Model Training
    # ---------------------------------------------------------
    print("\nStep 4: Training VectorRegressor...")

    model = VectorRegressor()

    # Double check parameters were applied (though config modification should handle it)
    for m in model.models:
        current_n = m.get_params().get("n_estimators")
        if current_n != 10:
            print(f"  Adjusting model params manually (found {current_n})...")
            m.set_params(**config.LGBM_PARAMS)

    model.fit(X_train, y_train, X_val, y_val)
    print("  Training complete.")

    # ---------------------------------------------------------
    # 5. Evaluation
    # ---------------------------------------------------------
    print("\nStep 5: Evaluating Model...")

    mae = model.evaluate(X_val, y_val)
    print(f"  Validation Mean Angular Error: {mae:.4f} rad")

    assert mae >= 0, "MAE cannot be negative"
    assert mae <= np.pi, "MAE cannot exceed Pi"

    # ---------------------------------------------------------
    # 6. Inference & Submission
    # ---------------------------------------------------------
    print("\nStep 6: Generating Submission...")

    # Predict unit vectors
    pred_vectors = model.predict(X_test)

    # Convert to spherical coordinates for submission
    pred_azimuth, pred_zenith = utils.cartesian_to_spherical(
        pred_vectors[:, 0], pred_vectors[:, 1], pred_vectors[:, 2]
    )

    # Create DataFrame
    submission_df = pd.DataFrame(
        {"event_id": ids_test, "azimuth": pred_azimuth, "zenith": pred_zenith}
    )

    # Sort by event_id as per sample submission (though not strictly required if IDs match)
    submission_df = submission_df.sort_values("event_id")

    # Verify format against sample
    sample_sub = pd.read_csv(config.SAMPLE_SUBMISSION_PATH, nrows=5)
    required_cols = list(sample_sub.columns)

    assert (
        list(submission_df.columns) == required_cols
    ), f"Submission columns mismatch. Expected {required_cols}, got {list(submission_df.columns)}"

    # Save to CSV
    output_path = config.SUBMISSION_DIR / "submission_demo.csv"
    submission_df.to_csv(output_path, index=False)

    print(f"  Submission saved to: {output_path}")
    print(f"  Submission head:\n{submission_df.head().to_string(index=False)}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
