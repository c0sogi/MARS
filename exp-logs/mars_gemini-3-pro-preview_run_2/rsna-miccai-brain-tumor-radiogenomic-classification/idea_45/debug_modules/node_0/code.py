import os
import sys
import torch
import pandas as pd
import numpy as np
import logging

# ------------------------------------------------------------------------------
# 1. Import Library Modules
# ------------------------------------------------------------------------------
import library.config as config
import library.utils as utils
import library.data_loader as data_loader
import library.model as model_lib
import library.train as train_lib
import library.predict as predict_lib

# ------------------------------------------------------------------------------
# 2. Configuration & Patching for Fast Demonstration
# ------------------------------------------------------------------------------
# We patch the imported constants in the library modules to ensure the demo
# runs quickly (1 epoch, small batch size) without modifying the source files.

DEMO_BATCH_SIZE = 4
DEMO_EPOCHS = 1
DEMO_PATIENCE = 1

# Patch data_loader
data_loader.BATCH_SIZE = DEMO_BATCH_SIZE

# Patch model library
model_lib.EPOCHS = DEMO_EPOCHS
model_lib.PATIENCE = DEMO_PATIENCE
model_lib.BATCH_SIZE = DEMO_BATCH_SIZE

# Patch train library
train_lib.EPOCHS = DEMO_EPOCHS
train_lib.PATIENCE = DEMO_PATIENCE

# Configure Logging to be less verbose for the demo
logging.getLogger("data_loader").setLevel(logging.WARNING)
logging.getLogger("model").setLevel(logging.WARNING)
logging.getLogger("train").setLevel(logging.WARNING)
logging.getLogger("predict").setLevel(logging.WARNING)

# ------------------------------------------------------------------------------
# 3. Main Demonstration Script
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Starting Demonstration of Glioblastoma Subtype Prediction Library ===\n")

    # Setup
    utils.seed_everything(config.SEED)
    config.setup_directories()
    device = config.get_device()
    print(f"Device: {device}")

    # --------------------------------------------------------------------------
    # Step 1: Verify Data Loading
    # --------------------------------------------------------------------------
    print("\n[1/5] Verifying Data Loading Pipeline...")

    # Use debug=True and max_samples to load a tiny subset
    train_loader, val_loader, test_loader = data_loader.get_dataloaders(
        debug=True, max_samples=16
    )

    # Fetch one batch
    try:
        images, labels = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("Train loader is empty!")

    print(f"   Batch Shapes -> Images: {images.shape}, Labels: {labels.shape}")

    # Assertions
    # Shape: (Batch, Channels, Height, Width) -> (4, 12, 224, 224)
    expected_channels = config.INPUT_CHANNELS  # 12
    expected_size = config.IMG_SIZE  # 224

    assert (
        images.shape[0] == DEMO_BATCH_SIZE
    ), f"Batch size mismatch. Expected {DEMO_BATCH_SIZE}, got {images.shape[0]}"
    assert (
        images.shape[1] == expected_channels
    ), f"Channel mismatch. Expected {expected_channels}, got {images.shape[1]}"
    assert (
        images.shape[2] == expected_size and images.shape[3] == expected_size
    ), "Image resolution mismatch."
    assert labels.shape[0] == DEMO_BATCH_SIZE, "Label batch size mismatch."
    assert images.dtype == torch.float32, "Image tensor should be float32."

    print("   Data Loader Verification Passed.")

    # --------------------------------------------------------------------------
    # Step 2: Verify Model Architecture
    # --------------------------------------------------------------------------
    print("\n[2/5] Verifying Asymmetric EfficientNet Model...")

    net = model_lib.AsymmetricEfficientNet().to(device)

    # Forward pass check
    inputs = images.to(device)
    outputs = net(inputs)

    print(f"   Output Shape: {outputs.shape}")

    # Assertions
    assert outputs.shape == (
        DEMO_BATCH_SIZE,
        1,
    ), "Model output shape should be (Batch_Size, 1)."
    assert not torch.isnan(outputs).any(), "Model produced NaN outputs."

    print("   Model Architecture Verification Passed.")

    # --------------------------------------------------------------------------
    # Step 3: Verify Training Logic (Trainer Class)
    # --------------------------------------------------------------------------
    print("\n[3/5] Verifying Trainer Logic (Single Epoch)...")

    trainer = train_lib.Trainer(net, train_loader, val_loader, device)

    # Run one training epoch
    train_loss, train_auc = trainer.train_epoch()
    print(f"   Train Epoch Result -> Loss: {train_loss:.4f}, AUC: {train_auc:.4f}")

    # Run validation
    val_loss, val_auc = trainer.validate()
    print(f"   Validation Result  -> Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

    # Assertions
    assert isinstance(train_loss, float), "Train loss should be a float."
    assert 0.0 <= train_auc <= 1.0, "Train AUC must be between 0 and 1."

    print("   Trainer Logic Verification Passed.")

    # --------------------------------------------------------------------------
    # Step 4: Verify Full Pipeline Execution
    # --------------------------------------------------------------------------
    print("\n[4/5] Verifying Full Training Pipeline...")

    # This runs the full loop: Data Load -> Init Model -> Train (1 epoch) -> Save -> Predict -> Submit
    train_lib.run_training_pipeline(debug=True, max_samples=20)

    # Check if submission file was created
    submission_path = config.SUBMISSION_PATH
    if not os.path.exists(submission_path):
        raise FileNotFoundError(
            f"Pipeline failed to generate submission file at {submission_path}"
        )

    print(f"   Submission file generated at: {submission_path}")
    print("   Pipeline Verification Passed.")

    # --------------------------------------------------------------------------
    # Step 5: Verify Prediction/Inference Standalone
    # --------------------------------------------------------------------------
    print("\n[5/5] Verifying Standalone Inference...")

    # This simulates the inference-only phase using the saved model from Step 4
    predict_lib.generate_submission(debug=True, max_samples=20)

    # Load and verify content format
    df_sub = pd.read_csv(submission_path)
    print(f"   Loaded Submission Shape: {df_sub.shape}")
    print(f"   Head:\n{df_sub.head(3)}")

    # Assertions
    required_cols = ["BraTS21ID", "MGMT_value"]
    assert all(
        col in df_sub.columns for col in required_cols
    ), f"Submission missing columns. Found: {df_sub.columns}"
    assert (
        df_sub["MGMT_value"].dtype == float or df_sub["MGMT_value"].dtype == np.float64
    ), "MGMT_value should be float."
    assert (
        df_sub["BraTS21ID"].dtype == int or df_sub["BraTS21ID"].dtype == np.int64
    ), "BraTS21ID should be int."

    print("   Inference Verification Passed.")

    print("\n=== Demonstration Completed Successfully ===")
