import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, check_initial_loss
from library.data import get_loaders
from library.model import AppleResNet34
from library.engine import fit, generate_submission


def main():
    print("==== Starting Demonstration Script ====")

    # 1. Setup and Configuration Override for Speed
    seed_everything(42)
    device = Config.DEVICE

    # Override Config for rapid demonstration
    Config.MAX_EPOCHS = 2
    print(f"Device: {device}")
    print(f"Max Epochs (Demo): {Config.MAX_EPOCHS}")

    # 2. Data Loading Demonstration & Verification
    print("\n[Step 1] Initializing Data Loaders...")

    # Get calibration loaders (Fold 0)
    train_loader, val_loader = get_loaders(fold_idx=0, phase="calibration")

    # Get test loader
    test_loader = get_loaders(phase="test")

    # Verify Train Batch
    images, targets = next(iter(train_loader))
    print(f"Train Batch - Images Shape: {images.shape}")
    print(f"Train Batch - Targets Shape: {targets.shape}")

    # Assertions to verify data integrity
    assert images.dim() == 4, "Images must be 4D tensors (B, C, H, W)"
    assert images.shape[1] == 3, "Images must have 3 channels"
    assert (
        targets.shape[1] == Config.NUM_CLASSES
    ), f"Targets must have {Config.NUM_CLASSES} classes"
    assert isinstance(images, torch.Tensor), "Images must be torch Tensors"
    assert isinstance(targets, torch.Tensor), "Targets must be torch Tensors"
    print("Data Loader verification passed.")

    # 3. Model Initialization & Initial Loss Check
    print("\n[Step 2] Initializing Model and Checking Initial Loss...")
    model = AppleResNet34()
    model.to(device)

    # Define criterion for the check (Standard CrossEntropy)
    # Note: The actual training uses weighted loss, but for initialization check
    # unweighted is sufficient to check for ~ln(1/4)
    criterion_check = torch.nn.CrossEntropyLoss()

    # Perform check
    initial_loss = check_initial_loss(model, train_loader, criterion_check, device)

    # Assert that loss is not zero and is a valid float
    assert initial_loss > 0, "Initial loss should be positive"
    assert isinstance(initial_loss, float), "Loss should be a float"
    print("Model initialization verification passed.")

    # 4. Training Loop Demonstration
    print("\n[Step 3] Running Training Loop (Calibration Phase)...")

    # Train for 2 epochs
    history, best_epoch = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=Config.MAX_EPOCHS,
        device=device,
        patience=2,  # Short patience for demo
    )

    # Verify History
    print(f"Best Epoch: {best_epoch}")
    print(f"Training History Keys: {history.keys()}")

    assert "train_loss" in history, "History must contain train_loss"
    assert "val_loss" in history, "History must contain val_loss"
    assert "val_auc" in history, "History must contain val_auc"
    assert (
        len(history["train_loss"]) == Config.MAX_EPOCHS
    ), "Should have logged loss for all epochs"
    print("Training loop verification passed.")

    # 5. Inference and Submission Demonstration
    print("\n[Step 4] Generating Submission...")

    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    generate_submission(
        model=model, test_loader=test_loader, device=device, output_path=submission_path
    )

    # Verify Submission File
    assert os.path.exists(submission_path), "Submission file was not created"

    df_sub = pd.read_csv(submission_path)
    print(f"Submission Shape: {df_sub.shape}")
    print(f"Submission Columns: {df_sub.columns.tolist()}")

    # Expected shape: 183 test images, 5 columns (image_id + 4 classes)
    expected_rows = 183
    expected_cols = 5

    assert (
        df_sub.shape[0] == expected_rows
    ), f"Expected {expected_rows} rows, got {df_sub.shape[0]}"
    assert (
        df_sub.shape[1] == expected_cols
    ), f"Expected {expected_cols} columns, got {df_sub.shape[1]}"
    assert "image_id" in df_sub.columns, "image_id column missing"
    assert all(
        col in df_sub.columns for col in Config.TARGET_COLS
    ), "Target columns missing"

    # Check values are probabilities (0-1)
    target_values = df_sub[Config.TARGET_COLS].values
    assert np.all(target_values >= 0) and np.all(
        target_values <= 1
    ), "Predictions must be probabilities between 0 and 1"

    print("Submission verification passed.")

    print("\n==== Demonstration Completed Successfully ====")


if __name__ == "__main__":
    main()
