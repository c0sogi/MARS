import os
import torch
import pandas as pd
import numpy as np
import warnings
from library.utils import set_seed, get_device
from library.data import get_dataloaders
from library.model import AsymmetricEfficientNet
from library.train import run_training

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("--- Starting Demonstration Script ---")

    # 1. Setup
    # Ensure reproducibility
    set_seed(42)
    device = get_device()
    print(f"Device selected: {device}")

    # Define parameters for a quick run
    DEBUG_LIMIT = 4  # Only use 4 samples per split
    BATCH_SIZE = 2
    EPOCHS = 1

    # 2. Verify Data Loading Logic
    print("\n[Verification] Testing Data Loading Pipeline...")

    # We force load_cached_data=False initially to test the raw DICOM processing logic
    # This will create cache files in ./working/idea_10/
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE, load_cached_data=False, debug_limit=DEBUG_LIMIT
    )

    # Fetch a single batch to verify shapes and types
    try:
        images, targets = next(iter(train_loader))
    except StopIteration:
        raise AssertionError("Train loader is empty!")

    print(f"Batch shapes - Images: {images.shape}, Targets: {targets.shape}")

    # Assertions for Data
    # Expected shape: (Batch, Channels=12, H=224, W=224)
    assert images.shape == (
        BATCH_SIZE,
        12,
        224,
        224,
    ), f"Incorrect image shape. Expected {(BATCH_SIZE, 12, 224, 224)}, got {images.shape}"

    # Expected shape: (Batch,)
    assert targets.shape == (
        BATCH_SIZE,
    ), f"Incorrect target shape. Expected {(BATCH_SIZE,)}, got {targets.shape}"

    # Check normalization (should be roughly 0-1, though augmentations might slightly shift)
    assert (
        images.max() <= 1.0 and images.min() >= 0.0
    ), "Images do not appear to be normalized to [0, 1] range."

    print("Data loading verification passed.")

    # 3. Verify Model Architecture
    print("\n[Verification] Testing Model Architecture...")

    model = AsymmetricEfficientNet(num_classes=1)
    model = model.to(device)

    # Perform a forward pass with the batch fetched earlier
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)

    print(f"Model output shape: {outputs.shape}")

    # Assertions for Model
    # Expected output: (Batch, 1) logits
    assert outputs.shape == (
        BATCH_SIZE,
        1,
    ), f"Incorrect model output shape. Expected {(BATCH_SIZE, 1)}, got {outputs.shape}"

    print("Model architecture verification passed.")

    # 4. Verify Full Training Pipeline
    print("\n[Verification] Testing Full Training Loop...")

    # We use run_training from library.train
    # We enable load_cached_data=True to use the cache we just generated in step 2
    run_training(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=1e-4,
        patience=1,
        debug_limit=DEBUG_LIMIT,
        load_cached_data=True,
    )

    print("Training loop completed.")

    # 5. Verify Submission Output
    print("\n[Verification] Checking Submission File...")

    submission_path = "./submission/submission.csv"

    if not os.path.exists(submission_path):
        raise FileNotFoundError(
            f"Submission file was not generated at {submission_path}"
        )

    df_sub = pd.read_csv(submission_path)
    print(f"Submission file loaded. Rows: {len(df_sub)}")
    print(df_sub.head())

    # Assertions for Submission
    expected_cols = ["BraTS21ID", "MGMT_value"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"

    # Since we used debug_limit=4, the test set should also be limited to 4
    assert (
        len(df_sub) == DEBUG_LIMIT
    ), f"Submission length mismatch. Expected {DEBUG_LIMIT}, got {len(df_sub)}"

    # Check values are probabilities (0-1)
    assert (
        df_sub["MGMT_value"].min() >= 0.0 and df_sub["MGMT_value"].max() <= 1.0
    ), "Submission values are not valid probabilities."

    print("Submission verification passed.")
    print("\n--- Demonstration Completed Successfully ---")


if __name__ == "__main__":
    main()
