import os
import sys
import pandas as pd
import torch
import numpy as np

# Import from the provided library files
from library.utils import seed_everything, get_device
from library.data_loader import get_dataloaders
from library.model import AsymmetricEfficientNet
from library.train_eval import run_training, generate_submission


def run_demonstration():
    print("=== Starting Demonstration ===")

    # 1. Setup
    seed_everything(42)
    device = get_device()
    print(f"Device: {device}")

    # Define paths
    working_dir = "./working"
    model_save_path = os.path.join(working_dir, "demo_model.pth")
    submission_path = os.path.join(working_dir, "demo_submission.csv")

    os.makedirs(working_dir, exist_ok=True)

    # 2. Demonstrate Data Loading
    print("\n--- Testing Data Loader ---")
    # We use a debug_limit to load only a few samples for speed
    batch_size = 4
    debug_limit = 12

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=2,
        load_cached_data=False,  # Force re-computation for demo purposes
        debug_limit=debug_limit,
    )

    print(f"Train loader length: {len(train_loader)}")

    # Fetch one batch to verify shapes
    try:
        images, labels = next(iter(train_loader))
        print(f"Batch Images Shape: {images.shape}")
        print(f"Batch Labels Shape: {labels.shape}")

        # Validation: Check shapes
        # Expected: (Batch, Channels=12, Height=224, Width=224)
        assert images.shape == (
            batch_size,
            12,
            224,
            224,
        ), f"Expected image shape {(batch_size, 12, 224, 224)}, got {images.shape}"
        assert labels.shape == (
            batch_size,
        ), f"Expected label shape {(batch_size,)}, got {labels.shape}"

        print("Data Loader verification passed.")
    except StopIteration:
        print("Error: Train loader is empty.")
        sys.exit(1)

    # 3. Demonstrate Model Instantiation & Forward Pass
    print("\n--- Testing Model Architecture ---")
    model = AsymmetricEfficientNet(pretrained=True).to(device)

    # Move batch to device
    images = images.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(images)

    print(f"Model Output Shape: {outputs.shape}")

    # Validation: Check output shape (Batch, 1)
    assert outputs.shape == (
        batch_size,
        1,
    ), f"Expected output shape {(batch_size, 1)}, got {outputs.shape}"

    print("Model architecture verification passed.")

    # 4. Demonstrate Training Loop
    print("\n--- Testing Training Loop ---")
    # Run for 2 epochs on the small debug subset
    best_auc = run_training(
        epochs=2,
        batch_size=batch_size,
        debug_limit=debug_limit,
        save_path=model_save_path,
        patience=2,
    )

    print(f"Training finished with Best AUC: {best_auc}")

    # Validation: Check if model file was created
    if not os.path.exists(model_save_path):
        # If AUC was 0 or validation failed to improve, it might not save.
        # However, with pretrained weights, it usually saves at least once or we force save.
        # The provided code saves if val_auc > best_auc (init 0.0).
        # If val_auc is 0.0, it won't save. Let's check if we need to handle that.
        # For demo, we assume at least one random guess > 0.0 or we manually save if needed.
        # But let's assert strictly to ensure the loop ran correctly.
        if best_auc > 0:
            assert os.path.exists(model_save_path), "Model checkpoint was not saved."
            print("Model checkpoint verified.")
        else:
            print(
                "Warning: Best AUC was 0.0, model might not have saved. Saving manually for inference step."
            )
            torch.save(model.state_dict(), model_save_path)

    # 5. Demonstrate Inference / Submission
    print("\n--- Testing Submission Generation ---")
    # Generate submission using the trained model
    # Note: generate_submission loads the full test set (59 samples), which is fast.
    generate_submission(
        model_path=model_save_path, output_path=submission_path, batch_size=batch_size
    )

    # Validation: Check submission file
    assert os.path.exists(submission_path), "Submission file was not created."

    df_sub = pd.read_csv(submission_path)
    print(f"Submission DataFrame Shape: {df_sub.shape}")
    print(f"Columns: {list(df_sub.columns)}")

    # Check columns
    assert (
        "BraTS21ID" in df_sub.columns and "MGMT_value" in df_sub.columns
    ), "Submission file missing required columns."

    # Check row count (Test set has 59 samples)
    # We read the metadata/test.csv to confirm expected length
    df_test_meta = pd.read_csv("./metadata/test.csv")
    expected_rows = len(df_test_meta)
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows in submission, got {len(df_sub)}"

    print("Submission verification passed.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demonstration()
