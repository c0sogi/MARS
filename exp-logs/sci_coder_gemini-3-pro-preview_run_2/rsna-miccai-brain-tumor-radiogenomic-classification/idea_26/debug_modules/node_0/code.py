import os
import torch
import pandas as pd
import numpy as np
import time
import sys

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloader, BraTSDataset
from library.model import AsymmetricEfficientNet
from library.train import run_training
from library.predict import predict_submission


def run_demonstration():
    print("=== Starting Demonstration of Asymmetric Grouped EfficientNet Pipeline ===")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    # Override Config for speed and demonstration purposes
    # Note: We modify class attributes directly. Since Python modules are cached,
    # these changes will be reflected when other modules access Config attributes at runtime.
    print("Configuring parameters for rapid demonstration...")
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 8  # Small batch size for quick iteration
    Config.NUM_WORKERS = 2

    # Ensure working directory is clean or exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds for reproducibility
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # --------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # --------------------------------------------------------------------------
    print("\n--- Verifying Data Pipeline ---")

    # Initialize DataLoader for Training
    # We use load_cached_data=True to utilize any pre-computed ROI indices
    train_loader = get_dataloader(
        "train", batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # Fetch a single batch to verify shapes and types
    try:
        images, targets = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("DataLoader is empty! Check metadata and input directory.")

    print(f"Batch loaded. Images shape: {images.shape}, Targets shape: {targets.shape}")

    # Assertions for Data Integrity
    # Expected shape: (Batch_Size, Input_Channels, Height, Width)
    # Input Channels = 24 (4 modalities * 2 strides * 3 slices)
    expected_channels = 24
    assert images.dim() == 4, f"Expected 4D tensor, got {images.dim()}D"
    assert (
        images.shape[1] == expected_channels
    ), f"Expected {expected_channels} channels, got {images.shape[1]}"
    assert (
        images.shape[2] == Config.IMG_SIZE and images.shape[3] == Config.IMG_SIZE
    ), f"Expected resolution {Config.IMG_SIZE}x{Config.IMG_SIZE}, got {images.shape[2]}x{images.shape[3]}"

    # Targets should be float32 for BCEWithLogitsLoss
    assert (
        targets.dtype == torch.float32
    ), f"Expected target dtype float32, got {targets.dtype}"

    print("Data Pipeline verification passed.")

    # --------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # --------------------------------------------------------------------------
    print("\n--- Verifying Model Architecture ---")

    model = AsymmetricEfficientNet().to(device)

    # Pass the sample batch through the model
    images = images.to(device)

    # Ensure model is in train mode to check gradient flow if needed (though we just check forward here)
    model.train()
    logits = model(images)

    print(f"Model forward pass successful. Output shape: {logits.shape}")

    # Assertions for Model Output
    # Output should be (Batch_Size, 1) for binary classification
    assert logits.shape == (
        images.size(0),
        1,
    ), f"Expected output shape {(images.size(0), 1)}, got {logits.shape}"

    # Check if gradients can be computed
    loss_fn = torch.nn.BCEWithLogitsLoss()
    targets = targets.to(device).unsqueeze(1)  # Match shape (B, 1)
    loss = loss_fn(logits, targets)
    loss.backward()

    # Verify that stem weights have gradients (ensuring the custom stem is connected)
    # The stem is at model.backbone.features[0][0]
    stem_layer = model.backbone.features[0][0]
    assert (
        stem_layer.weight.grad is not None
    ), "Gradients not computed for the custom stem layer."

    print("Model Architecture verification passed.")

    # --------------------------------------------------------------------------
    # 4. Training Loop Execution
    # --------------------------------------------------------------------------
    print("\n--- Executing Training Loop (1 Epoch) ---")

    # Run training using the library function
    # We explicitly pass num_epochs=1 to override the default argument in the function definition
    start_time = time.time()
    run_training(num_epochs=1, load_cached_data=True)
    end_time = time.time()

    print(f"Training finished in {end_time - start_time:.2f} seconds.")

    # Verify that the model checkpoint was saved
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.MODEL_SAVE_PATH} after training."
        )

    print(f"Checkpoint verified at: {Config.MODEL_SAVE_PATH}")

    # --------------------------------------------------------------------------
    # 5. Inference & Submission Generation
    # --------------------------------------------------------------------------
    print("\n--- Executing Inference & Submission ---")

    # Run prediction
    predict_submission(load_cached_data=True)

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH} after inference."
        )

    # Validate Submission Content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission loaded. Shape: {df_sub.shape}")
    print(df_sub.head())

    # Check columns
    expected_cols = ["BraTS21ID", "MGMT_value"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Expected columns {expected_cols}, got {list(df_sub.columns)}"

    # Check value range (probabilities)
    assert (
        df_sub["MGMT_value"].min() >= 0.0 and df_sub["MGMT_value"].max() <= 1.0
    ), "Predictions contain values outside [0, 1] range."

    # Check that we have predictions for the test set
    # Load test metadata to compare counts
    df_test_meta = pd.read_csv(Config.TEST_METADATA)
    assert len(df_sub) == len(
        df_test_meta
    ), f"Submission row count ({len(df_sub)}) does not match test set size ({len(df_test_meta)})."

    print("Inference verification passed.")
    print("\n=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    run_demonstration()
