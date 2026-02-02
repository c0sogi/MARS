import os
import numpy as np
import pandas as pd
import sys

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.data_loader import load_train_dataset, load_val_dataset, load_test_dataset
from library.model import DirectionalLGBM
from library.utils import cartesian_to_spherical, angular_dist_score
from library.config import SUBMISSION_PATH


def main():
    # ==========================================
    # 0. Setup and Configuration
    # ==========================================
    print("Initializing demonstration...")

    # Set seeds for reproducibility
    np.random.seed(42)

    # Define subset sizes for speed optimization
    TRAIN_ROWS = 5000
    VAL_ROWS = 1000
    TEST_ROWS = 1000

    # ==========================================
    # 1. Data Loading & Feature Engineering
    # ==========================================
    print(f"Loading training data (subset={TRAIN_ROWS})...")
    # We disable cache loading to demonstrate the feature generation pipeline
    X_train, y_train = load_train_dataset(
        load_cached_data=False, debug_n_rows=TRAIN_ROWS
    )

    print(f"Loading validation data (subset={VAL_ROWS})...")
    X_val, y_val = load_val_dataset(load_cached_data=False, debug_n_rows=VAL_ROWS)

    # --- Validation Checks ---
    print("Verifying data integrity...")
    # Check shapes
    assert len(X_train) == len(
        y_train
    ), "Mismatch in training features and targets length"
    assert len(X_val) == len(
        y_val
    ), "Mismatch in validation features and targets length"
    assert X_train.shape[1] > 0, "No features generated"
    assert y_train.shape[1] == 3, "Targets must be 3-dimensional (x, y, z)"

    # Check for NaNs
    assert not X_train.isnull().values.any(), "NaNs found in training features"
    assert not y_train.isnull().values.any(), "NaNs found in training targets"

    # Check target normalization (ground truth should be unit vectors)
    train_norms = np.linalg.norm(y_train.values, axis=1)
    assert np.allclose(
        train_norms, 1.0, atol=1e-5
    ), "Training targets are not unit vectors"

    print("Data loaded and verified successfully.")

    # ==========================================
    # 2. Model Initialization & Training
    # ==========================================
    print("Initializing model...")
    model = DirectionalLGBM()

    # Optimize hyperparameters for speed (Override config defaults)
    # Reducing estimators and increasing learning rate for quick demo convergence
    fast_params = {
        "n_estimators": 50,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "max_depth": 7,
        "verbose": -1,
    }

    for component in ["x", "y", "z"]:
        model.models[component].set_params(**fast_params)

    print("Training model...")
    model.fit(X_train, y_train, X_val, y_val)

    # ==========================================
    # 3. Evaluation
    # ==========================================
    print("Evaluating on validation set...")

    # Predict unit vectors
    val_preds_vec = model.predict(X_val)

    # --- Validation Checks ---
    # Check shape
    assert val_preds_vec.shape == (len(X_val), 3), "Prediction shape mismatch"
    # Check normalization
    pred_norms = np.linalg.norm(val_preds_vec, axis=1)
    assert np.allclose(pred_norms, 1.0, atol=1e-5), "Predictions are not unit vectors"

    # Convert predictions to spherical coordinates (azimuth, zenith)
    pred_az, pred_zen = cartesian_to_spherical(
        val_preds_vec[:, 0], val_preds_vec[:, 1], val_preds_vec[:, 2]
    )

    # Convert ground truth to spherical coordinates for metric calculation
    true_az, true_zen = cartesian_to_spherical(
        y_val["x"].values, y_val["y"].values, y_val["z"].values
    )

    # Calculate Metric
    y_true_spherical = {"azimuth": true_az, "zenith": true_zen}
    y_pred_spherical = {"azimuth": pred_az, "zenith": pred_zen}

    score = angular_dist_score(y_true_spherical, y_pred_spherical)
    print(f"Validation Mean Angular Error: {score:.4f} radians")

    # Sanity check on metric
    assert 0 <= score <= np.pi, "Angular error out of theoretical range [0, pi]"

    # ==========================================
    # 4. Submission Generation
    # ==========================================
    print(f"Loading test data (subset={TEST_ROWS})...")
    X_test, test_ids = load_test_dataset(load_cached_data=False, debug_n_rows=TEST_ROWS)

    print("Generating predictions for test set...")
    test_preds_vec = model.predict(X_test)

    # Convert to spherical for submission
    test_az, test_zen = cartesian_to_spherical(
        test_preds_vec[:, 0], test_preds_vec[:, 1], test_preds_vec[:, 2]
    )

    # Create submission DataFrame
    submission_df = pd.DataFrame(
        {"event_id": test_ids, "azimuth": test_az, "zenith": test_zen}
    )

    # --- Validation Checks ---
    assert len(submission_df) == TEST_ROWS, "Submission length mismatch"
    assert list(submission_df.columns) == [
        "event_id",
        "azimuth",
        "zenith",
    ], "Incorrect submission columns"
    assert (
        submission_df["azimuth"].min() >= 0
        and submission_df["azimuth"].max() <= 2 * np.pi
    ), "Azimuth out of range"
    assert (
        submission_df["zenith"].min() >= 0 and submission_df["zenith"].max() <= np.pi
    ), "Zenith out of range"

    # Save submission
    print(f"Saving submission to {SUBMISSION_PATH}...")
    # Ensure directory exists (handled by config, but good practice)
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(SUBMISSION_PATH, index=False)

    if os.path.exists(SUBMISSION_PATH):
        print("Submission file successfully created.")
    else:
        raise FileNotFoundError("Failed to save submission file.")

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
