import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, calculate_class_weights
from library.dataset import AppleDataset, get_transforms
from library.model import AppleResNet34
from library.engine import train_one_epoch, validate, predict_and_save


def run_demo():
    # 1. Setup and Reproducibility
    print("Step 1: Setting up environment and seeds...")
    seed_everything(Config.SEED)

    # Define a temporary working directory for this demo
    demo_working_dir = os.path.join(Config.WORKING_DIR, "demo_execution")
    os.makedirs(demo_working_dir, exist_ok=True)

    # 2. Data Loading (Using Subsets for Speed)
    print("\nStep 2: Loading metadata and creating datasets (Subset)...")

    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Create small subsets (e.g., 10 samples each) for demonstration
    train_subset = train_df.head(10).reset_index(drop=True)
    val_subset = val_df.head(10).reset_index(drop=True)
    test_subset = test_df.head(10).reset_index(drop=True)

    print(f"  Train subset shape: {train_subset.shape}")
    print(f"  Val subset shape: {val_subset.shape}")
    print(f"  Test subset shape: {test_subset.shape}")

    # Get transforms for Phase 1 resolution (256x256)
    train_transform = get_transforms(
        data_split="train", img_size=Config.IMG_SIZE_PHASE_1
    )
    val_transform = get_transforms(data_split="val", img_size=Config.IMG_SIZE_PHASE_1)

    # Instantiate Datasets
    train_dataset = AppleDataset(metadata=train_subset, transform=train_transform)
    val_dataset = AppleDataset(metadata=val_subset, transform=val_transform)
    # For test, we need output_extra=True to get image_ids for submission
    test_dataset = AppleDataset(
        metadata=test_subset, transform=val_transform, output_extra=True
    )

    # Instantiate DataLoaders
    batch_size = 2
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Verification: Check batch structure
    sample_batch = next(iter(train_loader))
    assert "image" in sample_batch, "DataLoader batch missing 'image' key"
    assert "target" in sample_batch, "DataLoader batch missing 'target' key"
    # Shape check: (Batch, Channels, Height, Width)
    assert sample_batch["image"].shape == (
        batch_size,
        3,
        Config.IMG_SIZE_PHASE_1,
        Config.IMG_SIZE_PHASE_1,
    ), f"Incorrect image shape: {sample_batch['image'].shape}"
    # Target check: (Batch, Num_Classes)
    assert sample_batch["target"].shape == (
        batch_size,
        Config.NUM_CLASSES,
    ), f"Incorrect target shape: {sample_batch['target'].shape}"
    print("  Dataset and DataLoader verification passed.")

    # 3. Model Initialization
    print("\nStep 3: Initializing Model...")
    # Using pretrained=False for speed and to avoid network dependency in this demo
    model = AppleResNet34(num_classes=Config.NUM_CLASSES, pretrained=False)
    model.to(Config.DEVICE)

    # Verification: Forward pass
    with torch.no_grad():
        dummy_input = torch.randn(
            batch_size, 3, Config.IMG_SIZE_PHASE_1, Config.IMG_SIZE_PHASE_1
        ).to(Config.DEVICE)
        dummy_output = model(dummy_input)
        assert dummy_output.shape == (
            batch_size,
            Config.NUM_CLASSES,
        ), f"Model output shape mismatch. Expected {(batch_size, Config.NUM_CLASSES)}, got {dummy_output.shape}"
    print("  Model initialization and forward pass verification passed.")

    # 4. Training Loop Demonstration
    print("\nStep 4: Running Training Loop (1 Epoch)...")
    # Setup training components
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # Using BCEWithLogitsLoss for multi-label/multi-class classification
    # We can optionally use class weights
    class_weights = calculate_class_weights(
        Config.TRAIN_METADATA_PATH, load_cached_data=False
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    # Train for 1 epoch
    train_loss = train_one_epoch(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=Config.DEVICE,
        epoch=1,
    )
    assert isinstance(train_loss, float), "Train loss should be a float"
    print(f"  Training completed. Loss: {train_loss:.4f}")

    # 5. Validation Demonstration
    print("\nStep 5: Running Validation...")
    val_loss, val_auc, val_preds, val_targets = validate(
        model=model, dataloader=val_loader, criterion=criterion, device=Config.DEVICE
    )

    assert isinstance(val_loss, float), "Validation loss should be a float"
    assert isinstance(val_auc, float), "Validation AUC should be a float"
    assert val_preds.shape == (
        len(val_subset),
        Config.NUM_CLASSES,
    ), "Validation predictions shape mismatch"
    print(f"  Validation completed. Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

    # 6. Inference and Submission
    print("\nStep 6: Running Inference and Generating Submission...")
    submission_path = os.path.join(demo_working_dir, "submission", "submission.csv")

    predict_and_save(
        model=model,
        dataloader=test_loader,
        device=Config.DEVICE,
        save_path=submission_path,
    )

    # Verification: Check submission file
    assert os.path.exists(submission_path), "Submission file was not created"

    sub_df = pd.read_csv(submission_path)
    print(f"  Submission file loaded. Shape: {sub_df.shape}")

    # Check columns
    expected_cols = ["image_id"] + Config.TARGET_COLS
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(sub_df.columns)}"

    # Check row count
    assert len(sub_df) == len(
        test_subset
    ), f"Submission row count mismatch. Expected {len(test_subset)}, got {len(sub_df)}"

    # Check values are probabilities (0-1) - allowing small float tolerance
    numeric_cols = Config.TARGET_COLS
    assert (sub_df[numeric_cols].values >= 0.0).all() and (
        sub_df[numeric_cols].values <= 1.0 + 1e-6
    ).all(), "Submission contains values outside [0, 1] probability range"

    print("  Submission verification passed.")
    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    run_demo()
