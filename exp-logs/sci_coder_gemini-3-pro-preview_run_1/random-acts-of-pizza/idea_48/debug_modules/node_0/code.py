import sys
import os
import numpy as np
import pandas as pd
import torch

# Ensure the library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything
from library.feature_engineering import FeatureEngineer
from library.trainer import train_rf_model, predict_rf, train_mlp_model, predict_mlp


def run_demo():
    print("=== Starting Demonstration Script ===")

    # 1. Configure Hyperparameters for Speed
    # We override the default configuration to ensure the demo runs quickly.
    print("Configuring hyperparameters for fast execution...")
    Config.RF_ESTIMATORS = 10  # Reduced from 500
    Config.RF_N_JOBS = 2  # Use 2 cores
    Config.NUM_EPOCHS = 2  # Reduced from 50
    Config.BATCH_SIZE = 32  # Standard batch size
    Config.PATIENCE = 2  # Reduced patience

    # Ensure reproducibility
    seed_everything(Config.RANDOM_SEED)

    # 2. Feature Engineering
    print("\n--- Step 1: Feature Engineering ---")
    # Instantiate the FeatureEngineer
    fe = FeatureEngineer()

    # Process data. We set load_cached_data=False to demonstrate the generation logic.
    # In a real scenario with large data, we would likely use True.
    print("Generating features (this may take a moment due to SBERT embeddings)...")
    rf_data, mlp_data = fe.process_data(load_cached_data=False)

    # Verification of Feature Engineering Output
    print("Verifying feature engineering outputs...")

    # Verify RF Data
    assert isinstance(rf_data, dict), "rf_data should be a dictionary"
    required_rf_keys = ["X_train", "y_train", "X_val", "y_val", "X_test"]
    for key in required_rf_keys:
        assert key in rf_data, f"rf_data missing key: {key}"

    # Check shapes
    n_train = rf_data["X_train"].shape[0]
    assert (
        rf_data["y_train"].shape[0] == n_train
    ), "RF X_train and y_train row counts mismatch"
    assert (
        rf_data["X_val"].shape[0] == rf_data["y_val"].shape[0]
    ), "RF X_val and y_val row counts mismatch"

    # Verify MLP Data
    assert isinstance(mlp_data, dict), "mlp_data should be a dictionary"
    required_mlp_keys = ["train", "val", "test"]
    for key in required_mlp_keys:
        assert key in mlp_data, f"mlp_data missing key: {key}"
        assert "metadata" in mlp_data[key], f"mlp_data[{key}] missing 'metadata'"

    print("Feature engineering verification successful.")

    # 3. Random Forest Pipeline
    print("\n--- Step 2: Random Forest Pipeline ---")

    # Train RF Model
    # force_retrain=True ensures we actually run the training code
    rf_model = train_rf_model(rf_data, force_retrain=True)

    # Generate Predictions
    print("Generating RF predictions...")
    val_probs_rf = predict_rf(rf_model, rf_data["X_val"])
    test_probs_rf = predict_rf(rf_model, rf_data["X_test"])

    # Verification of RF Predictions
    print("Verifying RF predictions...")
    assert len(val_probs_rf) == len(
        rf_data["y_val"]
    ), "RF val predictions length mismatch"
    assert (
        len(test_probs_rf) == rf_data["X_test"].shape[0]
    ), "RF test predictions length mismatch"

    # Check probability range [0, 1]
    assert np.all(
        (val_probs_rf >= 0.0) & (val_probs_rf <= 1.0)
    ), "RF val probs out of range"
    assert np.all(
        (test_probs_rf >= 0.0) & (test_probs_rf <= 1.0)
    ), "RF test probs out of range"

    print(f"RF Validation Predictions Sample: {val_probs_rf[:5]}")
    print("RF pipeline verification successful.")

    # 4. MLP Pipeline
    print("\n--- Step 3: MLP Pipeline ---")

    # Train MLP Model
    mlp_pipeline = train_mlp_model(mlp_data, force_retrain=True)

    # Generate Predictions
    # Note: predict_mlp handles DataLoader creation internally
    print("Generating MLP predictions...")
    val_probs_mlp = predict_mlp(mlp_pipeline, mlp_data, split="val")
    test_probs_mlp = predict_mlp(mlp_pipeline, mlp_data, split="test")

    # Verification of MLP Predictions
    print("Verifying MLP predictions...")
    # Get ground truth length for validation
    y_val_mlp = mlp_data["val"]["y"]
    if isinstance(y_val_mlp, np.ndarray) and y_val_mlp.ndim == 0:
        y_val_mlp = y_val_mlp.item()  # Handle 0-d array if loaded from npz weirdly

    assert len(val_probs_mlp) == len(y_val_mlp), "MLP val predictions length mismatch"

    # Check probability range [0, 1]
    assert np.all(
        (val_probs_mlp >= 0.0) & (val_probs_mlp <= 1.0)
    ), "MLP val probs out of range"
    assert np.all(
        (test_probs_mlp >= 0.0) & (test_probs_mlp <= 1.0)
    ), "MLP test probs out of range"

    print(f"MLP Validation Predictions Sample: {val_probs_mlp[:5]}")
    print("MLP pipeline verification successful.")

    # 5. Ensemble and Submission
    print("\n--- Step 4: Ensemble & Submission ---")

    # Simple weighted average ensemble
    ensemble_test_probs = (test_probs_rf * Config.RF_WEIGHT) + (
        test_probs_mlp * Config.MLP_WEIGHT
    )

    # Load Test Metadata to get IDs
    test_df = pd.read_csv(Config.TEST_PATH)

    # Create Submission DataFrame
    submission = pd.DataFrame(
        {Config.ID_COL: test_df[Config.ID_COL], Config.TARGET_COL: ensemble_test_probs}
    )

    # Verify Submission Format
    print("Verifying submission format...")
    assert submission.shape == (len(test_df), 2), "Submission shape mismatch"
    assert Config.ID_COL in submission.columns, f"Submission missing {Config.ID_COL}"
    assert (
        Config.TARGET_COL in submission.columns
    ), f"Submission missing {Config.TARGET_COL}"
    assert not submission.isnull().values.any(), "Submission contains NaN values"

    # Display sample
    print("Submission Head:")
    print(submission.head())

    # Save (Optional, but good practice to show where it goes)
    # We save to working directory for the demo
    demo_submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission.to_csv(demo_submission_path, index=False)
    print(f"Submission saved to {demo_submission_path}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
