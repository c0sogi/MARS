import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Import provided library modules
from library import config, data_loader, model, train_eval


def run_demo():
    print("Starting Demo Execution...")

    # ==========================================
    # 1. Configuration Override for Demo
    # ==========================================
    print("\n[1] Configuring environment for fast demo...")

    # Use a separate working directory for this demo to avoid conflicts
    demo_working_dir = "./working/demo_execution"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir, exist_ok=True)

    # Override config values
    config.WORKING_DIR = demo_working_dir
    config.SUBMISSION_DIR = demo_working_dir
    config.SUBMISSION_PATH = os.path.join(demo_working_dir, "submission.csv")
    config.NUM_EPOCHS = 1  # Run only 1 epoch
    config.BATCH_SIZE = 2  # Small batch size
    config.NUM_WORKERS = 2  # Reduce workers for small data

    print(f"Working Directory: {config.WORKING_DIR}")
    print(f"Epochs: {config.NUM_EPOCHS}, Batch Size: {config.BATCH_SIZE}")

    # ==========================================
    # 2. Model Logic Verification
    # ==========================================
    print("\n[2] Verifying Model Architecture...")

    # Instantiate model
    net = model.MGMTNet()
    net.eval()

    # Create dummy input: (Batch=2, Channels=128, H=256, W=256)
    # Channels = 4 modalities * 32 slices = 128
    dummy_input = torch.randn(2, 128, 256, 256)

    # Forward pass
    with torch.no_grad():
        output = net(dummy_input)

    # Check output shape (Batch, 1)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    if output.shape != (2, 1):
        raise AssertionError(f"Expected output shape (2, 1), got {output.shape}")

    print("Model verification passed.")

    # ==========================================
    # 3. Data Loader Verification
    # ==========================================
    print("\n[3] Verifying Data Loading Pipeline...")

    # Use a tiny sample size to force processing of just a few patients
    debug_size = 4

    # Get loaders (force no cache to test processing logic)
    train_loader, val_loader, test_loader = data_loader.get_dataloaders(
        load_cached_data=False, debug_sample_size=debug_size
    )

    # Fetch one batch from training loader
    images, labels = next(iter(train_loader))

    print(f"Train Batch Image Shape: {images.shape}")
    print(f"Train Batch Label Shape: {labels.shape}")

    # Verify shapes
    # Image: (Batch, 128, 256, 256)
    expected_img_shape = (config.BATCH_SIZE, 128, 256, 256)
    if images.shape != expected_img_shape:
        raise AssertionError(
            f"Expected images shape {expected_img_shape}, got {images.shape}"
        )

    # Label: (Batch,) - The dataloader returns 1D tensor for labels
    if labels.shape[0] != config.BATCH_SIZE:
        raise AssertionError(
            f"Expected {config.BATCH_SIZE} labels, got {labels.shape[0]}"
        )

    print("Data Loader verification passed.")

    # ==========================================
    # 4. Training Loop Execution
    # ==========================================
    print("\n[4] Executing Training Loop (1 Epoch, Subset)...")

    # Run training with the debug subset
    best_auc = train_eval.run_training(
        load_cached_data=True,  # It will load the cache we just created in step 3
        debug_sample_size=debug_size,
    )

    print(f"Training finished. Best AUC: {best_auc}")

    # Verify model checkpoint exists
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError("best_model.pth was not created!")

    print("Training execution passed.")

    # ==========================================
    # 5. Inference and Submission
    # ==========================================
    print("\n[5] Generating Submission...")

    # Generate submission using the trained model
    # Note: generate_submission uses the full test set defined in metadata
    train_eval.generate_submission(load_cached_data=False)

    # Verify submission file
    if not os.path.exists(config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {config.SUBMISSION_PATH}"
        )

    # Load and validate content
    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    print("Submission File Head:")
    print(df_sub.head())

    # Check columns
    required_cols = ["BraTS21ID", "MGMT_value"]
    if not all(col in df_sub.columns for col in required_cols):
        raise AssertionError(f"Submission missing columns. Found: {df_sub.columns}")

    # Check values
    if df_sub.isnull().values.any():
        raise AssertionError("Submission contains NaN values.")

    preds = df_sub["MGMT_value"]
    if preds.min() < 0 or preds.max() > 1:
        raise AssertionError("Predictions out of range [0, 1].")

    print(f"Submission generated with {len(df_sub)} rows.")
    print("Demo execution completed successfully.")


if __name__ == "__main__":
    # Ensure reproducibility
    train_eval.set_seed(42)

    try:
        run_demo()
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        sys.exit(1)
