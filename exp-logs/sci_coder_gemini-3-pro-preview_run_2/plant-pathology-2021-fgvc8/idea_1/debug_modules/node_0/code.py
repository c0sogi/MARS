import os
import sys
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import provided library modules
from library import config
from library import dataset
from library import model
from library import engine


def main():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    print("Initializing demonstration...")

    # Set device
    device = config.DEVICE
    print(f"Device: {device}")

    # Set seeds for reproducibility
    dataset.set_seed(config.SEED)

    # ==========================================
    # 2. Data Loading & Subsetting
    # ==========================================
    print("\nLoading metadata...")
    # Load full metadata
    train_df_full = pd.read_csv(config.TRAIN_META_PATH)
    val_df_full = pd.read_csv(config.VAL_META_PATH)
    test_df_full = pd.read_csv(config.TEST_META_PATH)

    # Create subsets for rapid demonstration (optimize for speed)
    # We use a small number of samples to ensure the training loop finishes quickly
    train_subset = train_df_full.sample(n=256, random_state=config.SEED).reset_index(
        drop=True
    )
    val_subset = val_df_full.sample(n=128, random_state=config.SEED).reset_index(
        drop=True
    )

    # We use the full test set to ensure the submission file is complete and valid
    test_df = test_df_full

    print(f"Training subset size: {len(train_subset)}")
    print(f"Validation subset size: {len(val_subset)}")
    print(f"Test set size: {len(test_df)}")

    # ==========================================
    # 3. Dataset & DataLoader Instantiation
    # ==========================================
    print("\nCreating Datasets and DataLoaders...")

    # Get transforms
    train_transforms = dataset.get_transforms(mode="train")
    val_transforms = dataset.get_transforms(mode="val")
    test_transforms = dataset.get_transforms(mode="test")

    # Instantiate Datasets
    train_ds = dataset.AppleDataset(
        train_subset, transforms=train_transforms, mode="train"
    )
    val_ds = dataset.AppleDataset(val_subset, transforms=val_transforms, mode="val")
    test_ds = dataset.AppleDataset(test_df, transforms=test_transforms, mode="test")

    # Instantiate DataLoaders
    # Using config.BATCH_SIZE (32)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Logic Verification: Check batch shapes
    print("Verifying DataLoader output...")
    example_images, example_targets = next(iter(train_loader))

    # Expected shape: (Batch, 3, H, W)
    assert example_images.shape == (
        config.BATCH_SIZE,
        3,
        config.IMG_SIZE,
        config.IMG_SIZE,
    ), f"Unexpected image shape: {example_images.shape}"

    # Expected shape: (Batch, Num_Classes)
    assert example_targets.shape == (
        config.BATCH_SIZE,
        config.NUM_CLASSES,
    ), f"Unexpected target shape: {example_targets.shape}"

    print("DataLoader verification passed.")

    # ==========================================
    # 4. Model Initialization
    # ==========================================
    print("\nInitializing Model...")
    net = model.AppleDiseaseModel(pretrained=True)
    net = net.to(device)

    # Logic Verification: Check Model Forward Pass
    print("Verifying Model forward pass...")
    with torch.no_grad():
        # Move example batch to device
        dummy_input = example_images.to(device)
        dummy_output = net(dummy_input)

    assert dummy_output.shape == (
        config.BATCH_SIZE,
        config.NUM_CLASSES,
    ), f"Unexpected model output shape: {dummy_output.shape}"

    print("Model verification passed.")

    # ==========================================
    # 5. Training Loop Demonstration
    # ==========================================
    print("\nStarting Training Loop (1 Epoch)...")

    # Define Optimizer and Scheduler
    optimizer = optim.AdamW(net.parameters(), lr=config.LEARNING_RATE)
    # T_max=1 because we are only running 1 epoch for demo
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)

    # Train model
    # We use 1 epoch to satisfy "Optimize for Speed"
    trained_model, best_f1 = engine.train_model(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=1,
        patience=1,
    )

    print(f"Training demonstration complete. Best Val F1: {best_f1:.4f}")

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    print("\nGenerating Submission...")

    engine.generate_submission(
        model=trained_model,
        test_loader=test_loader,
        device=device,
        output_path=config.SUBMISSION_PATH,
    )

    # Logic Verification: Check Submission File
    print("Verifying submission file...")
    if not os.path.exists(config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {config.SUBMISSION_PATH}"
        )

    sub_df = pd.read_csv(config.SUBMISSION_PATH)

    # Check row count
    assert len(sub_df) == len(
        test_df
    ), f"Submission row count mismatch. Expected {len(test_df)}, got {len(sub_df)}"

    # Check columns
    expected_cols = ["image", "labels"]
    assert all(
        col in sub_df.columns for col in expected_cols
    ), f"Missing columns in submission. Expected {expected_cols}, got {sub_df.columns.tolist()}"

    print(f"Submission verified. File saved to {config.SUBMISSION_PATH}")
    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
