import os
import sys
import numpy as np
import pandas as pd
import warnings
import shutil

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Import from provided library
from library.utils import seed_everything
from library.data_loader import load_dataset
from library.feature_generator import generate_features
from library.model_handler import DualStreamGBDT, generate_submission
from library.config import WORKING_DIR


def main():
    print("=== Starting Contact Detection Pipeline Demo ===")

    # 1. Setup
    seed_everything(42)

    # Define demo-specific paths and parameters
    demo_submission_path = os.path.join(WORKING_DIR, "demo_submission.csv")

    # Clean up working directory for this demo run to ensure fresh execution
    # (Optional, but good for a self-contained demo)
    if os.path.exists(WORKING_DIR):
        # We don't delete the whole dir as config might rely on it,
        # but we ensure our specific demo files don't conflict.
        pass

    # 2. Data Loading (Using small samples for speed)
    print("\n[Step 1] Loading Data...")

    # Load a small subset of training data (e.g., 2000 rows)
    # We set load_cached_data=False to demonstrate the raw loading logic
    train_df = load_dataset(mode="train", load_cached_data=False, sample_size=2000)

    # Load a small subset of validation data
    val_df = load_dataset(mode="validation", load_cached_data=False, sample_size=1000)

    # Load a small subset of test data
    test_df = load_dataset(mode="test", load_cached_data=False, sample_size=1000)

    print(f"Train subset shape: {train_df.shape}")
    print(f"Val subset shape: {val_df.shape}")
    print(f"Test subset shape: {test_df.shape}")

    # Validate data loading
    assert not train_df.empty, "Training dataframe is empty!"
    assert not val_df.empty, "Validation dataframe is empty!"
    assert not test_df.empty, "Test dataframe is empty!"

    # 3. Feature Engineering
    print("\n[Step 2] Generating Features...")

    # We use unique mode names to avoid conflicting with any existing caches
    # and force computation.
    train_data = generate_features(train_df, mode="train_demo", load_cached_data=False)
    val_data = generate_features(val_df, mode="val_demo", load_cached_data=False)
    test_data = generate_features(test_df, mode="test_demo", load_cached_data=False)

    # Validate Feature Structure
    for stream in ["stream_a", "stream_b"]:
        assert stream in train_data, f"Missing {stream} in train data"
        assert "X" in train_data[stream], f"Missing X in train {stream}"
        assert "y" in train_data[stream], f"Missing y in train {stream}"

        # Check that features were actually generated (columns > 0)
        n_features = train_data[stream]["X"].shape[1]
        print(f"  {stream} features: {n_features}")
        assert n_features > 0, f"No features generated for {stream}"

    # 4. Model Training
    print("\n[Step 3] Training DualStreamGBDT Model...")

    model = DualStreamGBDT()

    # OVERRIDE HYPERPARAMETERS FOR SPEED
    # The default config has 3000 estimators. We reduce this significantly for the demo.
    model.xgb_params["n_estimators"] = 10
    model.xgb_params["early_stopping_rounds"] = 2
    model.xgb_params["max_depth"] = 3
    # Disable verbosity
    model.xgb_params["verbosity"] = 0

    # Train the model
    model.train(train_data, val_data)

    # Validate model artifacts
    assert model.model_a is not None, "Stream A model failed to train"
    assert model.model_b is not None, "Stream B model failed to train"
    print(f"  Stream A Threshold: {model.thresh_a}")
    print(f"  Stream B Threshold: {model.thresh_b}")

    # 5. Inference & Submission
    print("\n[Step 4] Generating Submission...")

    # Generate submission file
    generate_submission(model, test_data, demo_submission_path)

    # 6. Final Validation
    print("\n[Step 5] Validating Output...")

    if not os.path.exists(demo_submission_path):
        raise FileNotFoundError(
            f"Submission file was not created at {demo_submission_path}"
        )

    submission_df = pd.read_csv(demo_submission_path)
    print(f"  Submission rows: {len(submission_df)}")
    print(f"  Submission columns: {submission_df.columns.tolist()}")

    # Check schema
    expected_cols = ["contact_id", "contact"]
    assert (
        list(submission_df.columns) == expected_cols
    ), f"Invalid columns: {submission_df.columns}"

    # Check values
    assert (
        submission_df["contact"].isin([0, 1]).all()
    ), "Predictions must be binary (0 or 1)"

    # Verify we have predictions for the test IDs provided
    # Note: Since we split streams, the output is a concatenation.
    # We check if the count matches the input test set size (minus any dropped rows if logic dictated,
    # but logic here preserves rows via split).
    # The test_data dictionary contains split IDs.
    total_test_ids = len(test_data["stream_a"]["ids"]) + len(
        test_data["stream_b"]["ids"]
    )
    assert (
        len(submission_df) == total_test_ids
    ), f"Mismatch in prediction count. Expected {total_test_ids}, got {len(submission_df)}"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
