import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library files
from library.utils import set_seed, get_device, Logger
from library.data import get_dataloaders
from library.model import DVSEModel
from library.train import run


def main():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    # Define paths for the demo execution
    WORKING_DIR = "./working/demo_execution"
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Initialize Logger
    logger = Logger(verbose=True)
    logger.section("Demo Script Initialization")

    # Set seed for reproducibility
    set_seed(42)

    # Check compute device
    device = get_device()

    # ==========================================
    # 2. Data Loading Demonstration
    # ==========================================
    logger.section("Data Loading Verification")

    # We use debug_limit=10 to load only 10 samples per split for speed.
    # We use a small batch size of 2.
    batch_size = 2
    debug_limit = 10

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=0,  # Use 0 workers for simple debugging to avoid multiprocessing overhead
        load_cached_data=False,  # Force processing from scratch for demonstration
        debug_limit=debug_limit,
    )

    # Verify Train Loader
    logger.log("Verifying Train Loader batch structure...")
    train_batch = next(iter(train_loader))
    images, labels = train_batch

    # Expected shape: (Batch, Channels, H, W)
    # Channels = 64 (4 modalities * 16 slices)
    # H, W = 256
    expected_shape = (batch_size, 64, 256, 256)

    logger.log(f"Train Images Shape: {images.shape}")
    logger.log(f"Train Labels Shape: {labels.shape}")

    assert (
        images.shape == expected_shape
    ), f"Expected train images shape {expected_shape}, got {images.shape}"
    assert labels.shape == (
        batch_size,
    ), f"Expected train labels shape {(batch_size,)}, got {labels.shape}"

    # Verify Test Loader (Dual View)
    logger.log("Verifying Test Loader batch structure...")
    test_batch = next(iter(test_loader))
    img_even, img_odd, patient_ids = test_batch

    logger.log(f"Test Even View Shape: {img_even.shape}")
    logger.log(f"Test Odd View Shape: {img_odd.shape}")

    assert img_even.shape == expected_shape, "Test Even view shape mismatch"
    assert img_odd.shape == expected_shape, "Test Odd view shape mismatch"
    assert len(patient_ids) == batch_size, "Test patient IDs length mismatch"

    # ==========================================
    # 3. Model Architecture Verification
    # ==========================================
    logger.section("Model Verification")

    model = DVSEModel(
        model_name="efficientnet_b0",
        pretrained=False,  # False for speed/no-download in demo, though True is used in production
        in_chans=64,
        num_classes=1,
    )
    model.to(device)
    model.eval()

    # Perform a forward pass with the dummy train batch
    logger.log("Performing forward pass...")
    with torch.no_grad():
        inputs = images.to(device)
        outputs = model(inputs)

    logger.log(f"Output Logits Shape: {outputs.shape}")

    # Expected output: (Batch, 1)
    assert outputs.shape == (
        batch_size,
        1,
    ), f"Expected output shape {(batch_size, 1)}, got {outputs.shape}"

    logger.log("Model forward pass successful.")

    # ==========================================
    # 4. Full Pipeline Execution (Train -> Val -> Predict)
    # ==========================================
    logger.section("Running Full Training Pipeline (Demo)")

    # Run the encapsulated training routine
    # We use minimal epochs and patience for speed
    run(
        epochs=2,
        patience=1,
        learning_rate=1e-4,
        batch_size=4,
        debug_limit=12,  # Slightly larger than batch size to ensure multiple batches
        save_path=MODEL_SAVE_PATH,
        submission_path=SUBMISSION_PATH,
    )

    # ==========================================
    # 5. Output Validation
    # ==========================================
    logger.section("Validating Submission Output")

    if not os.path.exists(SUBMISSION_PATH):
        raise FileNotFoundError(f"Submission file was not created at {SUBMISSION_PATH}")

    df_sub = pd.read_csv(SUBMISSION_PATH)
    logger.log(f"Loaded submission file with {len(df_sub)} rows.")

    # Check columns
    expected_cols = ["BraTS21ID", "MGMT_value"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Expected columns {expected_cols}, got {list(df_sub.columns)}"

    # Check value ranges
    probs = df_sub["MGMT_value"]
    if not probs.between(0, 1).all():
        raise ValueError("Predicted probabilities are out of range [0, 1]")

    logger.log("Submission file format is correct.")
    logger.log("Demo execution completed successfully.")


if __name__ == "__main__":
    main()
