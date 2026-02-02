import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import the provided library modules
from library import config
from library import feature_engineering
from library import dataset
from library import model
from library import training


def set_seeds(seed=42):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def test_feature_engineering():
    """Verifies feature generation logic."""
    print("\n=== Testing Feature Engineering ===")

    # Generate features for training split in debug mode
    # This will use a subset of metadata (head 5000)
    X, y, ids = feature_engineering.generate_features(
        split="train",
        load_cached_data=False,  # Force regeneration for verification
        debug=True,
    )

    print(f"Generated Feature Shape: {X.shape}")
    print(f"Generated Target Shape: {y.shape}")

    # Assertions
    assert len(X) == len(y), "Features and targets must have same length"
    assert len(X) == len(ids), "Features and IDs must have same length"
    assert not X.isnull().values.any(), "Feature matrix contains NaNs"
    assert (
        X.select_dtypes(include=[np.number]).shape[1] == X.shape[1]
    ), "All features must be numeric"

    # Check specific columns expected from kinematics
    expected_cols = ["dist_lag_0", "rel_speed_lag_0"]
    for col in expected_cols:
        assert col in X.columns, f"Missing expected kinematic column: {col}"

    print("Feature engineering verification passed.")
    return X.shape[1]


def test_dataset(input_dim):
    """Verifies the PyTorch Dataset class."""
    print("\n=== Testing NFLContactDataset ===")

    # Initialize dataset (will load the cache generated in previous step)
    ds = dataset.NFLContactDataset(split="train", load_cached_data=True, debug=True)

    assert len(ds) > 0, "Dataset should not be empty"

    # Test __getitem__
    sample = ds[0]

    # Verify keys
    assert "features" in sample
    assert "target" in sample
    assert "contact_id" in sample

    # Verify types and shapes
    feat = sample["features"]
    target = sample["target"]

    assert isinstance(feat, torch.Tensor), "Features must be a tensor"
    assert isinstance(target, torch.Tensor), "Target must be a tensor"
    assert (
        feat.shape[0] == input_dim
    ), f"Feature dimension mismatch. Expected {input_dim}, got {feat.shape[0]}"
    assert feat.dtype == torch.float32, "Features must be float32"

    print("Dataset verification passed.")


def test_model_architecture(input_dim):
    """Verifies Model forward pass."""
    print("\n=== Testing KinematicMLP Architecture ===")

    device = config.DEVICE
    net = model.KinematicMLP(input_dim).to(device)

    # Create dummy batch
    batch_size = 32
    dummy_input = torch.randn(batch_size, input_dim).to(device)

    # Forward pass
    output = net(dummy_input)

    # Assertions
    assert output.shape == (
        batch_size,
        1,
    ), f"Output shape mismatch. Expected {(batch_size, 1)}, got {output.shape}"
    assert (
        output.min() >= 0 and output.max() <= 1
    ), "Output must be sigmoid probability (0-1)"

    print("Model architecture verification passed.")


def test_pipeline_execution():
    """Runs the training and prediction pipeline."""
    print("\n=== Testing Full Pipeline Execution ===")

    # 1. Train
    # training.train returns the best threshold
    best_threshold = training.train(debug=True)

    print(f"Training finished. Optimal threshold: {best_threshold}")
    assert 0.0 <= best_threshold <= 1.0, "Threshold must be between 0 and 1"
    assert os.path.exists(config.MODEL_SAVE_PATH), "Model checkpoint was not saved"

    # 2. Predict
    # training.predict generates submission file
    training.predict(best_threshold, debug=True)

    assert os.path.exists(
        config.SUBMISSION_FILE_PATH
    ), "Submission file was not created"

    # 3. Verify Submission File
    df_sub = pd.read_csv(config.SUBMISSION_FILE_PATH)
    print(f"Submission file loaded. Rows: {len(df_sub)}")

    assert "contact_id" in df_sub.columns, "Submission missing contact_id column"
    assert "contact" in df_sub.columns, "Submission missing contact column"
    assert df_sub["contact"].isin([0, 1]).all(), "Predictions must be binary (0 or 1)"

    print("Pipeline execution verification passed.")


if __name__ == "__main__":
    # 1. Setup
    set_seeds(42)

    # 2. Monkey-patch Config for Speed
    # We reduce epochs and model size to ensure the demo finishes quickly
    print("Configuring environment for rapid demonstration...")
    config.EPOCHS = 2
    config.BATCH_SIZE = 1024
    config.HIDDEN_LAYERS = [64, 32]  # Smaller model
    config.PATIENCE = 1

    # Ensure working directory is clean for features (optional, but good for verification)
    # Note: We don't delete metadata, only cached parquet files if we want a fresh start.
    # For this script, we let feature_engineering handle overwrite via load_cached_data=False initially.

    try:
        # 3. Test Components
        input_dim = test_feature_engineering()
        test_dataset(input_dim)
        test_model_architecture(input_dim)

        # 4. Run Pipeline
        test_pipeline_execution()

        print("\nAll demonstrations completed successfully.")

    except AssertionError as e:
        print(f"\n[FAILED] Assertion failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
