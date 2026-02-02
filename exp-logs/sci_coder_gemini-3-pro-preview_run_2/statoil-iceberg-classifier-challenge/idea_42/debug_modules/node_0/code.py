import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import from the provided library files
import library.config as config
from library.utils import seed_everything, calculate_log_loss
from library.data_loader import load_data
from library.model import DN_WBN
from library.train import train_one_epoch, validate, predict_test


def run_demo():
    print("=== Starting Library Demonstration ===\n")

    # 1. Setup and Configuration
    # --------------------------
    # Set seed for reproducibility
    seed_everything(config.SEED)

    # Determine device
    device = config.DEVICE
    print(f"Device: {device}")

    # Define demo parameters for speed
    DEMO_BATCH_SIZE = 8
    DEMO_DATA_LIMIT = 32  # Use only 32 samples for the demo
    DEMO_EPOCHS = 2

    # 2. Data Loading Demonstration
    # -----------------------------
    print("\n[Step 1] Demonstrating Data Loading...")

    # We force reload to ensure we don't just load a massive cached file if we want to be quick,
    # although the library handles slicing after loading.
    # We use the debug_limit parameter to get a small subset.
    train_loader, val_loader, test_loader = load_data(
        load_cached_data=True, batch_size=DEMO_BATCH_SIZE, debug_limit=DEMO_DATA_LIMIT
    )

    # Verify DataLoaders
    try:
        # Get a single batch from train_loader
        images, angles, labels = next(iter(train_loader))

        print(f"  Train Batch - Images Shape: {images.shape}")
        print(f"  Train Batch - Angles Shape: {angles.shape}")
        print(f"  Train Batch - Labels Shape: {labels.shape}")

        # Assertions
        assert images.shape == (
            DEMO_BATCH_SIZE,
            3,
            75,
            75,
        ), "Incorrect image batch shape"
        assert angles.shape == (DEMO_BATCH_SIZE,), "Incorrect angle batch shape"
        assert labels.shape == (DEMO_BATCH_SIZE,), "Incorrect label batch shape"

        # Check Normalization (approximate check since we don't know exact global min/max here easily)
        # But we know standard images shouldn't be huge if normalized.
        # The library does (img - min) / (max - min), so values should be roughly 0-1.
        print(f"  Image Value Range: [{images.min():.4f}, {images.max():.4f}]")

        print("  Data Loading Verification: PASSED")

    except StopIteration:
        raise Exception("DataLoader is empty!")
    except AssertionError as e:
        raise Exception(f"Data Loading Verification FAILED: {e}")

    # 3. Model Instantiation and Forward Pass
    # ---------------------------------------
    print("\n[Step 2] Demonstrating Model Architecture...")

    model = DN_WBN().to(device)
    print("  Model DN_WBN instantiated.")

    # Create dummy input based on batch shape
    dummy_img = torch.randn(DEMO_BATCH_SIZE, 3, 75, 75).to(device)
    dummy_ang = torch.randn(DEMO_BATCH_SIZE).to(device)

    # Forward pass
    try:
        output = model(dummy_img, dummy_ang)
        print(f"  Model Output Shape: {output.shape}")

        # Assertions
        assert output.shape == (DEMO_BATCH_SIZE,), "Model output shape mismatch"
        assert torch.all(output >= 0) and torch.all(
            output <= 1
        ), "Model output not in [0, 1] range (Sigmoid check)"

        print("  Model Forward Pass Verification: PASSED")

    except Exception as e:
        raise Exception(f"Model Verification FAILED: {e}")

    # 4. Training Loop Component Demonstration
    # ----------------------------------------
    print("\n[Step 3] Demonstrating Training Loop Components...")

    # Setup Optimizer and Criterion
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    print(f"  Running training for {DEMO_EPOCHS} epochs...")

    for epoch in range(DEMO_EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_metric = validate(model, val_loader, criterion, device)

        print(
            f"  Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, LogLoss={val_metric:.4f}"
        )

        # Assertions
        assert not np.isnan(train_loss), "Training loss is NaN"
        assert not np.isnan(val_loss), "Validation loss is NaN"
        assert isinstance(train_loss, float), "Train loss is not a float"

    print("  Training Loop Verification: PASSED")

    # 5. Prediction Demonstration
    # ---------------------------
    print("\n[Step 4] Demonstrating Prediction on Test Set...")

    try:
        ids, preds = predict_test(model, test_loader, device)

        print(f"  Predictions Generated: {len(preds)}")
        print(f"  Sample IDs: {ids[:3]}")
        print(f"  Sample Preds: {preds[:3]}")

        # Assertions
        # Note: Test loader might be smaller than DEMO_DATA_LIMIT if the test set is small,
        # but here we limited it.
        expected_len = min(len(test_loader.dataset), DEMO_DATA_LIMIT)
        assert (
            len(preds) == expected_len
        ), f"Expected {expected_len} predictions, got {len(preds)}"
        assert len(ids) == len(preds), "Mismatch between IDs and Predictions count"
        assert np.all(
            (preds >= 0) & (preds <= 1)
        ), "Predictions out of probability range"

        print("  Prediction Verification: PASSED")

    except Exception as e:
        raise Exception(f"Prediction Verification FAILED: {e}")

    # 6. Submission File Generation
    # -----------------------------
    print("\n[Step 5] Demonstrating Submission File Creation...")

    submission_df = pd.DataFrame({"id": ids, "is_iceberg": preds})

    # Save to a temporary location in working directory
    demo_submission_path = os.path.join(config.WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(demo_submission_path, index=False)

    print(f"  Submission saved to: {demo_submission_path}")

    # Verify file exists and format
    assert os.path.exists(demo_submission_path), "Submission file was not created"

    df_check = pd.read_csv(demo_submission_path)
    assert list(df_check.columns) == ["id", "is_iceberg"], "Submission columns mismatch"
    assert len(df_check) == len(preds), "Submission row count mismatch"

    print("  Submission File Verification: PASSED")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
