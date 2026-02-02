import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, mixup_data, mixup_criterion
from library.dataset import CadenceDataset, get_dataloaders
from library.model import Shallow3DCNN
from library.engine import fit


def main():
    print("=== Starting Pipeline Verification Script ===")

    # 1. Setup and Configuration Override
    # We override Config values to run a fast check
    print("\n[1] Setting up configuration for fast execution...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Small subset for speed
    Config.BATCH_SIZE = 8
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 2

    # Ensure reproducibility
    set_seed(Config.SEED)
    Config.setup()
    print("    Configuration updated: DEBUG=True, EPOCHS=1, BATCH_SIZE=8")

    # 2. Verify Dataset Logic
    print("\n[2] Verifying CadenceDataset...")
    # Load a small slice of metadata manually for unit testing the dataset class
    df_train = pd.read_csv(Config.TRAIN_METADATA).head(10)
    dataset = CadenceDataset(df_train, Config.INPUT_DIR, img_size=Config.IMG_SIZE)

    # Fetch one item
    image, target = dataset[0]

    # Check shapes
    # Expected shape: (1, 6, 256, 256) -> (Channel, Depth, Height, Width)
    # The dataset class returns (1, 6, 256, 256) because it treats the 6 cadence positions
    # as depth for the 3D CNN, but keeps a singleton channel dim.
    print(f"    Single item shape: {image.shape}")
    print(f"    Target value: {target}")

    assert image.ndim == 4, f"Expected 4 dimensions (C, D, H, W), got {image.ndim}"
    assert image.shape == (
        1,
        6,
        256,
        256,
    ), f"Expected (1, 6, 256, 256), got {image.shape}"
    assert isinstance(target, torch.Tensor), "Target should be a tensor"
    print("    Dataset verification passed.")

    # 3. Verify DataLoaders
    print("\n[3] Verifying DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        train_batch_size=Config.BATCH_SIZE,
        val_batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        debug=True,
        debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
    )

    # Get a batch
    images, targets = next(iter(train_loader))

    print(f"    Batch image shape: {images.shape}")
    print(f"    Batch target shape: {targets.shape}")

    # Expected: (Batch, 1, 6, 256, 256)
    assert images.shape == (
        Config.BATCH_SIZE,
        1,
        6,
        256,
        256,
    ), f"Expected ({Config.BATCH_SIZE}, 1, 6, 256, 256), got {images.shape}"
    assert targets.shape == (
        Config.BATCH_SIZE,
    ), f"Expected ({Config.BATCH_SIZE},), got {targets.shape}"
    print("    DataLoader verification passed.")

    # 4. Verify Model Architecture
    print("\n[4] Verifying Shallow3DCNN Model...")
    device = Config.DEVICE
    model = Shallow3DCNN().to(device)

    # Move batch to device
    images = images.to(device)

    # Forward pass
    outputs = model(images)

    print(f"    Model output shape: {outputs.shape}")

    # Expected output: (Batch, 1) (Logits)
    assert outputs.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Expected ({Config.BATCH_SIZE}, 1), got {outputs.shape}"
    print("    Model architecture verification passed.")

    # 5. Verify Mixup Utilities
    print("\n[5] Verifying Mixup Augmentation...")
    targets = targets.to(device).unsqueeze(1)  # (B, 1)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Apply mixup
    mixed_images, y_a, y_b, lam = mixup_data(images, targets, alpha=0.2, device=device)

    # Check mixed shape
    assert mixed_images.shape == images.shape, "Mixed images shape mismatch"

    # Calculate loss
    # We need to re-run the model on mixed images to get predictions for loss calculation
    mixed_outputs = model(mixed_images)
    loss = mixup_criterion(criterion, mixed_outputs, y_a, y_b, lam)

    print(f"    Mixup Loss: {loss.item()}")
    assert not torch.isnan(loss), "Mixup loss is NaN"
    print("    Mixup verification passed.")

    # 6. Verify Full Engine Execution (Fit)
    print("\n[6] Running Full Training Loop (Engine)...")

    # We use the loaders generated in step 3 which are already subsetted (debug mode)
    # We run for 1 epoch as configured above
    try:
        fit(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=Config.EPOCHS,
            device=Config.DEVICE,
        )
        print("    Engine execution completed without errors.")
    except Exception as e:
        print(f"    Engine execution failed: {e}")
        raise e

    # 7. Verify Submission Output
    print("\n[7] Verifying Submission File...")
    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"    Submission file found at {Config.SUBMISSION_PATH}")
        print(f"    Submission shape: {df_sub.shape}")
        print(f"    Columns: {list(df_sub.columns)}")

        # In debug mode, we subsetted the test set to DEBUG_SAMPLE_SIZE
        # So the submission should have DEBUG_SAMPLE_SIZE rows
        assert (
            len(df_sub) == Config.DEBUG_SAMPLE_SIZE
        ), f"Expected {Config.DEBUG_SAMPLE_SIZE} rows, got {len(df_sub)}"
        assert (
            "id" in df_sub.columns and "target" in df_sub.columns
        ), "Missing required columns in submission"

        # Check values are probabilities (sigmoid was applied in predict_with_tta)
        assert (
            df_sub["target"].min() >= 0.0 and df_sub["target"].max() <= 1.0
        ), "Predictions are not valid probabilities [0, 1]"

        print("    Submission verification passed.")
    else:
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    print("\n=== All Verifications Passed Successfully ===")


if __name__ == "__main__":
    main()
