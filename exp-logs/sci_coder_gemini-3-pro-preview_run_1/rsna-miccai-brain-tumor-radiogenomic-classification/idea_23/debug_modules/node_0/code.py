import os
import sys
import torch
import pandas as pd
import numpy as np

# =============================================================================
# 1. Configuration & Patching
# =============================================================================
# We import the config module first and patch it to enable DEBUG mode and
# reduce runtime parameters (epochs, dataset size) before importing other modules.
import library.config as config

print("Patching configuration for rapid demonstration...")
config.DEBUG = True
config.DEBUG_SAMPLE_SIZE = 40  # Small subset for speed
config.NUM_EPOCHS = 1  # Single epoch for demonstration
config.BATCH_SIZE = 8  # Smaller batch size
config.NUM_WORKERS = 2  # Reduce worker overhead
config.PATIENCE = 1  # Minimal patience

# Ensure the working directory for this run exists
os.makedirs(config.WORKING_DIR, exist_ok=True)

# =============================================================================
# 2. Imports from Library
# =============================================================================
# Now we import the rest of the library components. They will pick up the
# patched config values.
from library.utils import seed_everything, get_device
from library.data_loader import get_dataloaders
from library.network import build_model
from library.engine import train_model, generate_submission_csv

# =============================================================================
# 3. Main Execution
# =============================================================================
if __name__ == "__main__":
    # Set random seed for reproducibility
    seed_everything(config.SEED)

    device = get_device()
    print(f"Running on device: {device}")

    # -------------------------------------------------------------------------
    # Step 1: Data Loading
    # -------------------------------------------------------------------------
    print("\n--- Step 1: Initializing DataLoaders ---")
    # We force reload_cached_data=False to ensure the data processing logic
    # is exercised during this run, rather than loading old pre-computed arrays.
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        config.TRAIN_METADATA_PATH,
        config.VAL_METADATA_PATH,
        config.TEST_METADATA_PATH,
        load_cached_data=False,
    )

    # Validation: Check DataLoader outputs
    print("Validating DataLoader shapes...")
    # Fetch one batch
    images, targets = next(iter(train_loader))

    # Expected Image Shape: (Batch_Size, 9, IMG_SIZE, IMG_SIZE)
    # The 9 channels come from 3 modalities * 3 depth slices
    expected_shape = (
        config.BATCH_SIZE,
        config.IN_CHANNELS,
        config.IMG_SIZE,
        config.IMG_SIZE,
    )

    if images.shape != expected_shape:
        raise AssertionError(
            f"Incorrect input shape. Expected {expected_shape}, got {images.shape}"
        )

    # Expected Target Shape: (Batch_Size,)
    if targets.shape[0] != config.BATCH_SIZE:
        raise AssertionError(
            f"Incorrect target batch size. Expected {config.BATCH_SIZE}, got {targets.shape[0]}"
        )

    print(f"Data loading verified. Batch shape: {images.shape}")

    # -------------------------------------------------------------------------
    # Step 2: Model Initialization
    # -------------------------------------------------------------------------
    print("\n--- Step 2: Building AGIV Model ---")
    model = build_model(device)

    # Validation: Check Model Output Shape
    print("Validating model forward pass...")
    dummy_input = torch.randn(expected_shape).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    # Expected Output: (Batch_Size, 1) - Logits for binary classification
    if output.shape != (config.BATCH_SIZE, 1):
        raise AssertionError(
            f"Incorrect model output shape. Expected {(config.BATCH_SIZE, 1)}, got {output.shape}"
        )

    print("Model architecture verified.")

    # -------------------------------------------------------------------------
    # Step 3: Training Loop
    # -------------------------------------------------------------------------
    print("\n--- Step 3: Executing Training Loop ---")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Train the model (using the modified config.NUM_EPOCHS=1)
    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=config.NUM_EPOCHS,
        patience=config.PATIENCE,
    )

    # Validation: Check if model checkpoint was created
    if not os.path.exists(config.MODEL_SAVE_PATH):
        # Note: If validation AUC is 0.5 (random) and doesn't improve, it might not save "best".
        # However, engine.py saves if val_auc > best_val_auc (init 0.0).
        # Even a random model usually gets > 0.0 AUC.
        print(
            "Warning: No best model checkpoint found (Validation AUC might have been 0.0)."
        )
    else:
        print(f"Model checkpoint verified at {config.MODEL_SAVE_PATH}")

    # -------------------------------------------------------------------------
    # Step 4: Inference & Submission
    # -------------------------------------------------------------------------
    print("\n--- Step 4: Generating Submission ---")
    generate_submission_csv(
        trained_model, test_loader, test_ids, device, config.SUBMISSION_PATH
    )

    # Validation: Verify Submission File
    if not os.path.exists(config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not created at {config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    print(f"Submission file loaded. Rows: {len(df_sub)}")

    # Check columns
    expected_cols = ["BraTS21ID", "MGMT_value"]
    if list(df_sub.columns) != expected_cols:
        raise AssertionError(
            f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"
        )

    # Check if we have predictions for the test IDs
    # Note: In DEBUG mode, we only processed a subset of test IDs, so we check against `test_ids`
    if len(df_sub) != len(test_ids):
        raise AssertionError(
            f"Submission row count {len(df_sub)} does not match test set size {len(test_ids)}"
        )

    print("\n=== Demonstration Completed Successfully ===")
