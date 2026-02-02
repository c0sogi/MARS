import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import from the provided library
from library.config import Config
from library.utils import set_seed, setup_logger
from library.data import get_dataloaders
from library.model import PDPH_SE_CNN
from library.train import train_one_epoch, validate, predict


def demo_usage():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n=== 1. Configuration & Setup ===")

    # Override Config for speed and demonstration purposes
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 64  # Small subset for quick execution
    Config.BATCH_SIZE = 8
    Config.NUM_EPOCHS = 1
    Config.N_FOLDS = 2  # Not used in this linear demo, but good practice
    Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure working directory exists (Config creates it on import, but we verify)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print(f"Device: {Config.DEVICE}")
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n=== 2. Data Loading Verification ===")

    # Get dataloaders (this will process data if not cached in Config.WORKING_DIR)
    # Since we changed Config.WORKING_DIR or it's empty, it might process from scratch.
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Verify Train Loader
    print("Verifying Train Loader batch structure...")
    images, angles, labels = next(iter(train_loader))

    # Assertions for shapes
    # Images: (Batch, Channels, Height, Width) -> (8, 3, 75, 75)
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        75,
        75,
    ), f"Expected image shape {(Config.BATCH_SIZE, 3, 75, 75)}, got {images.shape}"

    # Angles: (Batch,) -> (8,)
    assert angles.shape == (
        Config.BATCH_SIZE,
    ), f"Expected angles shape {(Config.BATCH_SIZE,)}, got {angles.shape}"

    # Labels: (Batch,) -> (8,)
    assert labels.shape == (
        Config.BATCH_SIZE,
    ), f"Expected labels shape {(Config.BATCH_SIZE,)}, got {labels.shape}"

    print("  [OK] Batch shapes verified.")
    print(f"  Image statistics: Mean={images.mean():.4f}, Std={images.std():.4f}")

    # -------------------------------------------------------------------------
    # 3. Model Instantiation & Forward Pass
    # -------------------------------------------------------------------------
    print("\n=== 3. Model Logic Verification ===")

    model = PDPH_SE_CNN().to(Config.DEVICE)
    print(f"Model {model.__class__.__name__} instantiated.")

    # Move batch to device
    images = images.to(Config.DEVICE)
    angles = angles.to(Config.DEVICE)

    # Forward pass
    logits = model(images, angles)

    # Verify output shape: (Batch,)
    assert logits.shape == (
        Config.BATCH_SIZE,
    ), f"Expected output shape {(Config.BATCH_SIZE,)}, got {logits.shape}"

    # Verify output values are finite (no NaNs)
    assert torch.isfinite(logits).all(), "Model output contains NaNs or Infs"

    print("  [OK] Forward pass successful. Output shape matches batch size.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Simulation
    # -------------------------------------------------------------------------
    print("\n=== 4. Training Loop Simulation ===")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    print("Running training for 1 epoch...")
    train_loss = train_one_epoch(
        model, train_loader, criterion, optimizer, Config.DEVICE
    )
    print(f"  Train Loss: {train_loss:.6f}")

    # Assert loss is valid
    assert train_loss > 0, "Training loss should be positive"

    print("Running validation...")
    val_loss, val_preds, val_targets = validate(
        model, val_loader, criterion, Config.DEVICE
    )
    print(f"  Val Loss: {val_loss:.6f}")

    # Assertions for validation outputs
    assert len(val_preds) == len(
        val_targets
    ), "Mismatch between preds and targets count"
    assert len(val_preds) == len(
        val_loader.dataset
    ), "Predictions count matches dataset size"
    # Check probability range
    assert (val_preds >= 0).all() and (
        val_preds <= 1
    ).all(), "Predictions not in [0, 1] range"

    print("  [OK] Training and Validation functions executed successfully.")

    # -------------------------------------------------------------------------
    # 5. Inference & Submission Generation
    # -------------------------------------------------------------------------
    print("\n=== 5. Inference & Submission ===")

    print("Generating predictions for test set...")
    test_preds = predict(model, test_loader, Config.DEVICE)

    # Verify prediction count matches test set size (which is subsetted in DEBUG mode)
    assert len(test_preds) == len(
        test_loader.dataset
    ), f"Expected {len(test_loader.dataset)} predictions, got {len(test_preds)}"

    print(f"  Generated {len(test_preds)} predictions.")

    # Construct Submission DataFrame
    # We need IDs from the dataset. The loader returns (img, angle, id)
    # But predict() only returns preds. We access ids directly from the dataset.
    test_ids = test_loader.dataset.ids

    submission_df = pd.DataFrame({"id": test_ids, "is_iceberg": test_preds})

    print("Sample submission rows:")
    print(submission_df.head())

    # Verify against sample submission format
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION)
    expected_cols = list(sample_sub.columns)
    current_cols = list(submission_df.columns)

    assert (
        expected_cols == current_cols
    ), f"Column mismatch. Expected {expected_cols}, got {current_cols}"

    print("  [OK] Submission format verified.")

    # Save dummy submission
    out_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(out_path, index=False)
    print(f"Demo submission saved to {out_path}")


if __name__ == "__main__":
    try:
        demo_usage()
        print("\nSUCCESS: All demonstrations and verifications passed.")
    except AssertionError as e:
        print(f"\nFAILURE: Assertion failed - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nFAILURE: An unexpected error occurred - {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
