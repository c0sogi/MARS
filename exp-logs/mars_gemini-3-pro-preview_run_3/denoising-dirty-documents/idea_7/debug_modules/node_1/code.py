import os
import shutil
import torch
import pandas as pd
import numpy as np
import time

# Import from the provided library
from library.config import Config
from library import utils, model, data_loader, train, inference


def run_demo():
    print("--- Starting Demonstration of Denoising Pipeline ---")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup for Fast Execution
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Define a separate working directory for this demo to avoid conflicts
    DEMO_WORKING_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_WORKING_DIR):
        shutil.rmtree(DEMO_WORKING_DIR)
    os.makedirs(DEMO_WORKING_DIR, exist_ok=True)

    # Override Config parameters to speed up the process
    # We modify the class attributes directly so they propagate to other modules
    Config.WORKING_DIR = DEMO_WORKING_DIR
    Config.SUBMISSION_DIR = DEMO_WORKING_DIR
    Config.SUBMISSION_PATH = os.path.join(DEMO_WORKING_DIR, "demo_submission.csv")

    # Reduce Model Complexity for speed
    Config.NUM_RDN_BLOCKS = 2  # Default 8
    Config.NUM_LAYERS_PER_BLOCK = 2  # Default 4
    Config.GROWTH_RATE = 16  # Default 32
    Config.NUM_FEATURES = 16  # Default 64

    # Data Processing settings for speed
    Config.PATCH_SIZE = 40
    Config.STRIDE = 40  # Non-overlapping patches to reduce count
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Main process only for simplicity

    # Training settings
    Config.NUM_EPOCHS = 2
    Config.EARLY_STOPPING_PATIENCE = 2

    print(f"Working Directory set to: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Prepare Mini-Metadata
    # -------------------------------------------------------------------------
    print("\n[2] Creating mini-datasets for testing...")

    # Read original metadata
    orig_train = pd.read_csv(Config.TRAIN_METADATA)
    orig_val = pd.read_csv(Config.VAL_METADATA)
    orig_test = pd.read_csv(Config.TEST_METADATA)

    # Create subsets (few samples each)
    mini_train = orig_train.head(4).copy()
    mini_val = orig_val.head(2).copy()
    mini_test = orig_test.head(2).copy()

    # Save mini metadata
    mini_train_path = os.path.join(DEMO_WORKING_DIR, "mini_train.csv")
    mini_val_path = os.path.join(DEMO_WORKING_DIR, "mini_val.csv")
    mini_test_path = os.path.join(DEMO_WORKING_DIR, "mini_test.csv")

    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    # Point Config to these new files
    Config.TRAIN_METADATA = mini_train_path
    Config.VAL_METADATA = mini_val_path
    Config.TEST_METADATA = mini_test_path

    print(
        f"Mini-metadata created. Train: {len(mini_train)}, Val: {len(mini_val)}, Test: {len(mini_test)}"
    )

    # -------------------------------------------------------------------------
    # 3. Verify Data Loading and Processing
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Data Loading and Patch Extraction...")

    # Force processing (load_cached_data=False) to test the extraction logic with our new mini metadata
    train_loader, val_loader = data_loader.get_dataloaders(load_cached_data=False)

    # Check Train Loader
    train_batch = next(iter(train_loader))
    inputs, targets = train_batch

    print(f"Train Batch Shape: {inputs.shape}")

    # Assertions
    # Note: If total patches < BATCH_SIZE, the last batch might be smaller.
    # With 4 images and 40x40 patches, we should have plenty for a batch of 4.
    assert inputs.shape == (
        Config.BATCH_SIZE,
        1,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    ), f"Expected input shape {(Config.BATCH_SIZE, 1, Config.PATCH_SIZE, Config.PATCH_SIZE)}, got {inputs.shape}"
    assert targets.shape == inputs.shape, "Target shape mismatch"
    assert inputs.max() <= 1.0 and inputs.min() >= 0.0, "Input normalization failed"

    print("Data Loader verification successful.")

    # -------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    # Instantiate model
    net = model.SRDN().to(Config.DEVICE)

    # Create dummy input
    dummy_input = torch.randn(2, 1, Config.PATCH_SIZE, Config.PATCH_SIZE).to(
        Config.DEVICE
    )

    # Forward pass
    with torch.no_grad():
        output = net(dummy_input)

    print(f"Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == dummy_input.shape, "Model output shape mismatch"
    assert not torch.isnan(output).any(), "Model produced NaN values"

    print("Model architecture verification successful.")

    # -------------------------------------------------------------------------
    # 5. Verify Training Loop
    # -------------------------------------------------------------------------
    print("\n[5] Executing Training Loop (Short Run)...")

    # Run training for limited epochs/batches
    # We use load_cached_data=True because we generated the cache in step 3
    trained_model = train.run_training(
        load_cached_data=True, max_epochs=2, max_batches_per_epoch=5
    )

    # Check if best model file exists
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model checkpoint was not saved"

    print("Training loop execution successful.")

    # -------------------------------------------------------------------------
    # 6. Verify Inference and Submission
    # -------------------------------------------------------------------------
    print("\n[6] Executing Inference and Submission Generation...")

    # Generate submission using the trained model
    inference.generate_submission_file(
        model_path=best_model_path,
        output_path=Config.SUBMISSION_PATH,
        device=Config.DEVICE,
    )

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file created with {len(df_sub)} rows.")

    # Check header and content
    assert list(df_sub.columns) == ["id", "value"], "Submission columns mismatch"
    assert (
        df_sub["value"].min() >= 0 and df_sub["value"].max() <= 1
    ), "Submission values out of range"

    # Check ID format (e.g., '110_1_1')
    if len(df_sub) > 0:
        sample_id = df_sub.iloc[0]["id"]
        parts = sample_id.split("_")
        assert len(parts) == 3, f"Invalid ID format: {sample_id}"

    print("Inference and submission verification successful.")

    # -------------------------------------------------------------------------
    # 7. Verify Utility Functions
    # -------------------------------------------------------------------------
    print("\n[7] Verifying Utility Functions...")

    # Test RMSE
    t1 = torch.tensor([0.0, 0.5, 1.0])
    t2 = torch.tensor([0.0, 0.5, 1.0])
    rmse_val = utils.calculate_rmse(t1, t2)
    assert rmse_val.item() == 0.0, "RMSE check failed for identical tensors"

    t3 = torch.tensor([1.0])
    t4 = torch.tensor([0.0])
    rmse_val_2 = utils.calculate_rmse(t3, t4)
    # RMSE of 1 and 0 is 1.
    assert abs(rmse_val_2.item() - 1.0) < 1e-6, "RMSE check failed for diff 1.0"

    print("Utility verification successful.")

    print("\n--- Demonstration Complete: All checks passed ---")


if __name__ == "__main__":
    run_demo()
