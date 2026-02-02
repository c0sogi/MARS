import os
import shutil
import numpy as np
import pandas as pd
import torch

# Import provided library modules
import library.config as config
import library.utils as utils
import library.data as data
import library.model as model
from library.train import Trainer


def run_demo():
    print("=== Starting Demo Execution ===")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Define a separate directory for demo artifacts to ensure isolation
    DEMO_ROOT = "./working/demo_execution"
    if os.path.exists(DEMO_ROOT):
        shutil.rmtree(DEMO_ROOT)
    os.makedirs(DEMO_ROOT, exist_ok=True)

    DEMO_CACHE = os.path.join(DEMO_ROOT, "cache")
    DEMO_CHECKPOINTS = os.path.join(DEMO_ROOT, "checkpoints")
    DEMO_SUBMISSION_DIR = os.path.join(DEMO_ROOT, "submission")
    DEMO_SUBMISSION_FILE = os.path.join(DEMO_SUBMISSION_DIR, "submission.csv")

    # Create directories
    os.makedirs(DEMO_CACHE, exist_ok=True)
    os.makedirs(DEMO_CHECKPOINTS, exist_ok=True)
    os.makedirs(DEMO_SUBMISSION_DIR, exist_ok=True)

    print(f"Demo working directory: {DEMO_ROOT}")

    # Monkey-patch library modules to use demo paths and settings
    # This is necessary because the modules import variables directly

    # Update library.data settings
    data.WORKING_DIR = DEMO_CACHE  # Data module uses WORKING_DIR for caching
    data.DEBUG = True
    data.DEBUG_SAMPLES = 100  # Small subset for speed

    # Update library.model settings
    model.CHECKPOINT_DIR = DEMO_CHECKPOINTS
    model.SUBMISSION_PATH = DEMO_SUBMISSION_FILE
    model.EPOCHS = 1  # 1 Epoch for speed

    # Set seed for reproducibility
    utils.set_seed(42)

    # -------------------------------------------------------------------------
    # 2. Data Loading Verification
    # -------------------------------------------------------------------------
    print("\n[1/4] Verifying Data Loading...")

    # Force processing from raw JSON (load_cached_data=False) to test processing logic
    # This will use the DEBUG_SAMPLES limit set above
    (X_train, angles_train, y_train, ids_train), (X_test, angles_test, ids_test) = (
        data.load_data(load_cached_data=False)
    )

    # Verify shapes
    print(f"  Train Data Shape: {X_train.shape}")
    print(f"  Test Data Shape: {X_test.shape}")

    # Assertions
    assert (
        len(X_train) == data.DEBUG_SAMPLES
    ), f"Expected {data.DEBUG_SAMPLES} training samples, got {len(X_train)}"
    # Image shape: (N, 3, 75, 75)
    assert X_train.shape[1:] == (
        3,
        75,
        75,
    ), f"Unexpected image dimensions: {X_train.shape[1:]}"
    assert len(y_train) == len(X_train), "Label count mismatch"
    assert not np.isnan(X_train).any(), "Training data contains NaNs"

    # Verify angles
    # Angles can be NaN in raw data, but load_data returns raw angles.
    # Imputation happens in get_loaders.
    print("  Data loading verification passed.")

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[2/4] Verifying Model Architecture...")

    device = utils.get_device()
    net = model.RTICNN().to(device)

    # Create dummy input batch
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 75, 75).to(device)
    dummy_angles = torch.tensor([35.0, 40.0, 0.0, 32.5]).to(device)

    # Forward pass
    net.eval()
    with torch.no_grad():
        output = net(dummy_images, dummy_angles)

    print(f"  Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (
        batch_size,
    ), f"Expected output shape ({batch_size},), got {output.shape}"
    assert not torch.isnan(output).any(), "Model produced NaN outputs"
    print("  Model architecture verification passed.")

    # -------------------------------------------------------------------------
    # 4. Training Pipeline Verification
    # -------------------------------------------------------------------------
    print("\n[3/4] Verifying Training Pipeline...")

    # Initialize Trainer with debug=True (redundant given our patches, but good practice)
    # epochs=1 ensures it runs quickly
    trainer = Trainer(epochs=1, debug=True)

    # Run training
    trainer.train()

    # Verify checkpoints
    print("  Checking for checkpoints...")
    expected_checkpoints = [f"model_fold_{i}.pth" for i in range(config.NUM_FOLDS)]
    for ckpt_name in expected_checkpoints:
        ckpt_path = os.path.join(DEMO_CHECKPOINTS, ckpt_name)
        assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"

    print("  Training pipeline verification passed.")

    # -------------------------------------------------------------------------
    # 5. Inference Verification
    # -------------------------------------------------------------------------
    print("\n[4/4] Verifying Inference and Submission...")

    # Generate submission
    trainer.generate_submission()

    # Verify file existence
    assert os.path.exists(DEMO_SUBMISSION_FILE), "Submission file was not created"

    # Verify file content
    df_sub = pd.read_csv(DEMO_SUBMISSION_FILE)
    print(f"  Submission Head:\n{df_sub.head(3)}")

    # Assertions
    assert list(df_sub.columns) == [
        "id",
        "is_iceberg",
    ], "Submission columns are incorrect"
    assert len(df_sub) == len(
        ids_test
    ), f"Submission row count mismatch. Expected {len(ids_test)}, got {len(df_sub)}"

    # Check probability range
    probs = df_sub["is_iceberg"]
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Probabilities are out of [0, 1] range"

    print("  Inference verification passed.")

    print("\n=== Demo Execution Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
