import os
import pandas as pd
import numpy as np
import xgboost as xgb
from library.data_loader import NFLDataLoader
from library.model import DualStreamXGB
from library.utils import seed_everything
from library.config import SEED, SUBMISSION_PATH


def main():
    # 1. Setup
    print("--- Setting up environment ---")
    seed_everything(SEED)

    # 2. Data Loading
    # The NFLDataLoader handles feature engineering, caching, and splitting into Stream A/B
    print("\n--- Initializing Data Loader ---")
    loader = NFLDataLoader()

    print("Loading Training Data (this may take time for feature engineering)...")
    train_data = loader.prepare_streams(split="train")

    print("Loading Validation Data...")
    val_data = loader.prepare_streams(split="validation")

    # Verify Data Structure
    for stream in ["stream_a", "stream_b"]:
        assert stream in train_data, f"Missing {stream} in train_data"
        assert "X" in train_data[stream]
        assert "y" in train_data[stream]
        assert len(train_data[stream]["X"]) == len(train_data[stream]["y"])
        print(f"Loaded {stream} training samples: {len(train_data[stream]['X'])}")

    # 3. Data Subsampling (Optimization for Demo Speed)
    # We slice the training data to a smaller subset to ensure the script runs quickly
    DEMO_SAMPLE_SIZE = 10000
    print(
        f"\n--- Subsampling Training Data to {DEMO_SAMPLE_SIZE} samples for demo speed ---"
    )

    for stream in ["stream_a", "stream_b"]:
        n_samples = len(train_data[stream]["X"])
        if n_samples > DEMO_SAMPLE_SIZE:
            train_data[stream]["X"] = train_data[stream]["X"].iloc[:DEMO_SAMPLE_SIZE]
            train_data[stream]["y"] = train_data[stream]["y"][:DEMO_SAMPLE_SIZE]
            train_data[stream]["ids"] = train_data[stream]["ids"][:DEMO_SAMPLE_SIZE]
            print(f"Reduced {stream} to {len(train_data[stream]['X'])} samples.")

    # 4. Model Initialization & Configuration
    print("\n--- Initializing Model ---")
    model = DualStreamXGB()

    # Override hyperparameters for speed (Demo purposes)
    # The default n_estimators is 2000, which is too slow for a quick demo.
    print("Configuring model hyperparameters for fast execution (n_estimators=10)...")
    model.model_a.set_params(n_estimators=10, early_stopping_rounds=None)
    model.model_b.set_params(n_estimators=10, early_stopping_rounds=None)

    # 5. Training
    print("\n--- Starting Training ---")
    model.fit(train_data, val_data)

    # 6. Threshold Optimization
    print("\n--- Optimizing Thresholds ---")
    model.optimize_thresholds(val_data)

    # 7. Inference
    print("\n--- Loading Test Data & Generating Submission ---")
    test_data = loader.prepare_streams(split="test")

    model.generate_submission(test_data)

    # 8. Final Validation
    print("\n--- Validating Submission ---")
    if not os.path.exists(SUBMISSION_PATH):
        raise FileNotFoundError(f"Submission file not found at {SUBMISSION_PATH}")

    df_sub = pd.read_csv(SUBMISSION_PATH)
    print(f"Submission loaded. Shape: {df_sub.shape}")
    print(f"Columns: {df_sub.columns.tolist()}")

    # Validate submission length matches test set size
    expected_len = len(test_data["stream_a"]["ids"]) + len(test_data["stream_b"]["ids"])
    if len(df_sub) != expected_len:
        raise AssertionError(
            f"Submission length mismatch! Expected {expected_len}, got {len(df_sub)}"
        )

    # Validate content
    if not all(col in df_sub.columns for col in ["contact_id", "contact"]):
        raise AssertionError("Submission missing required columns.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
